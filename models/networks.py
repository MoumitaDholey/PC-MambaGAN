import functools

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.optim import lr_scheduler

from . import modules


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net):
    print("initialization method [normal]")
    net.apply(weights_init_normal)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Constant learning rate for opt.niter epochs, then linear decay to zero over opt.niter_decay epochs."""

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def define_G(input_nc, output_nc, gpu_ids=[]):
    """PC-MambaGAN generator. All parameters are initialised from N(0, 0.02)."""
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert torch.cuda.is_available()

    netG = modules.I2IMamba(input_dim=input_nc, output_dim=output_nc)
    for param in netG.parameters():
        if param.requires_grad:
            init.normal_(param.data, mean=0, std=0.02)

    if use_gpu:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, norm="instance", use_sigmoid=False, gpu_ids=[]):
    """Conditional PatchGAN discriminator with 3 layers."""
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert torch.cuda.is_available()

    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
                               use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically the same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input.numel()
            )
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input.numel()
            )
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        gpu_ids=[],
    ):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input, return_features=False):
        if not return_features:
            if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
                return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            else:
                return self.model(input)
        else:
            # Return every layer's activation (used by the feature-matching loss).
            features = []
            x = input
            for layer in self.model:
                x = layer(x)
                features.append(x)
            return features

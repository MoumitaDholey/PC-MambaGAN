import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchinfo import summary
from pytorch_msssim import ms_ssim
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import SimpleITK as sitk
import numpy as np
from radiomics import glcm
from .glhcm_loss import GLHCMLoss

# pyradiomics settings for the GLHCM texture features
settings = {
    "binWidth": 25,
    "label": 1,
    "interpolator": sitk.sitkBSpline,
    "resampledPixelSpacing": None,
    "normalize": True,
    "normalizeScale": 1,
}

hessianoption = 3  # 0: No Filter, 1: Frobenius norm, 2: Smallest eigenvalue, 3: Use both Hessian maps


class I2IMamba_model(BaseModel):
    """PC-MambaGAN: generator, conditional PatchGAN discriminator and the composite loss
    L_G = L_GAN + lambda_1 L_L1 + lambda_MS L_MS-SSIM + lambda_FM L_FM + lambda_GLHCM L_GLHCM."""

    def name(self):
        return "I2IMamba_model"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.norm,
                use_sigmoid,
                self.gpu_ids,
            )

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, "G", opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, "D", opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor
            )
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            # loss weights (lambda_1 = opt.lambda_A)
            self.lambda_MS_SSIM = 10.0
            self.lambda_FM = 5.0
            self.lambda_glcm = 5.0
            #self.criterionGLHCM = GLHCMLoss().cuda(self.gpu_ids[0]) if self.gpu_ids else GLHCMLoss()

        print("---------- Networks initialized -------------")
        summary(self.netG, input_size=(1, opt.input_nc, 256, 256))
        if self.isTrain:
            networks.print_network(self.netD)
        print("-----------------------------------------------")

    def set_input(self, input):
        AtoB = self.opt.which_direction == "AtoB"
        input_A = input["A" if AtoB else "B"]
        input_B = input["B" if AtoB else "A"]
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_adv
        self.loss_D.backward()

    def backward_G(self):
        # --- Adversarial Loss ---
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_adv

        # --- L1 Loss ---
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        # --- MS-SSIM, feature-matching and GLHCM texture losses ---
        fake_B_norm = (self.fake_B + 1) / 2.0
        real_B_norm = (self.real_B + 1) / 2.0
        self.loss_G_MS_SSIM = (1 - ms_ssim(fake_B_norm, real_B_norm, data_range=1.0)) * self.lambda_MS_SSIM

        fake_features = self.netD(fake_AB, return_features=True)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_features = self.netD(real_AB, return_features=True)
        fm_loss = 0.0
        for f_fake, f_real in zip(fake_features, real_features):
            fm_loss += self.criterionL1(f_fake, f_real)
        self.loss_G_FM = fm_loss * self.lambda_FM

        #self.glcm_loss = self.criterionGLHCM(self.fake_B, self.real_B) * self.lambda_glcm
        #extra_loss = self.loss_G_MS_SSIM + self.loss_G_FM + self.glcm_loss


        if hessianoption == 3:
            img_np_float = self.real_B[0, 0, :, :].detach().cpu().numpy().astype(np.float32)
            timage1, timage2 = self.heassianFilter(img_np_float)

            timage = sitk.GetImageFromArray(timage1)
            tmask = sitk.GetImageFromArray(np.ones_like(img_np_float, dtype=np.uint8))
            tmask.CopyInformation(timage)
            feat_real1 = self.glcm_texture(timage, tmask, settings)

            timage = sitk.GetImageFromArray(timage2)
            tmask = sitk.GetImageFromArray(np.ones_like(img_np_float, dtype=np.uint8))
            tmask.CopyInformation(timage)
            feat_real2 = self.glcm_texture(timage, tmask, settings)
            feat_real = self.average_features(feat_real1, feat_real2)

            img_np_float = self.fake_B[0, 0, :, :].detach().cpu().numpy().astype(np.float32)
            timage1, timage2 = self.heassianFilter(img_np_float)

            timage = sitk.GetImageFromArray(timage1)
            tmask = sitk.GetImageFromArray(np.ones_like(img_np_float, dtype=np.uint8))
            tmask.CopyInformation(timage)
            feat_fake1 = self.glcm_texture(timage, tmask, settings)

            timage = sitk.GetImageFromArray(timage2)
            tmask = sitk.GetImageFromArray(np.ones_like(img_np_float, dtype=np.uint8))
            tmask.CopyInformation(timage)
            feat_fake2 = self.glcm_texture(timage, tmask, settings)
            feat_fake = self.average_features(feat_fake1, feat_fake2)

        else:
            img_np_float = self.real_B[0, 0, :, :].detach().cpu().numpy().astype(np.float32)
            timage = self.heassianFilter(img_np_float)
            timage = sitk.GetImageFromArray(timage)
            tmask = sitk.GetImageFromArray(np.ones_like(img_np_float, dtype=np.uint8))
            tmask.CopyInformation(timage)
            feat_real = self.glcm_texture(timage, tmask, settings)

            img_np_float = self.fake_B[0, 0, :, :].detach().cpu().numpy().astype(np.float32)
            timage = self.heassianFilter(img_np_float)
            timage = sitk.GetImageFromArray(timage)
            tmask = sitk.GetImageFromArray(np.ones_like(img_np_float, dtype=np.uint8))
            tmask.CopyInformation(timage)
            feat_fake = self.glcm_texture(timage, tmask, settings)

        self.glcm_loss = self.sum_absolute_diff(feat_real, feat_fake) * self.lambda_glcm
        extra_loss = self.loss_G_MS_SSIM + self.loss_G_FM + self.glcm_loss

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + extra_loss
        self.loss_G.backward()

    def heassianFilter(self, image):
        if hessianoption == 0:
            return image
        else:
            sigma = 2.0
            H_elems = hessian_matrix(image, sigma=sigma, order='rc')
            eigvals = hessian_matrix_eigvals(H_elems)
            if hessianoption == 1:
                filter_image = np.sqrt(H_elems[0] ** 2 + 2 * H_elems[1] ** 2 + H_elems[2] ** 2)
                filter_image = self.to_uint8(filter_image)
            elif hessianoption == 2:
                filter_image = np.min(eigvals, axis=0)
                filter_image = self.to_uint8(filter_image)
            elif hessianoption == 3:
                filter_image1 = np.sqrt(H_elems[0] ** 2 + 2 * H_elems[1] ** 2 + H_elems[2] ** 2)
                filter_image2 = np.min(eigvals, axis=0)
                #filter_image = [self.to_uint8(filter_image1), self.to_uint8(filter_image2)]
                filter_image = [filter_image1, filter_image2]
            return filter_image

    def to_uint8(self, arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        return scaled.astype(np.uint8)

    def average_features(self, dict1, dict2):
        avg_dict = {}
        for key, val1 in dict1.items():
            if key in dict2:
                val2 = dict2[key]
                try:
                    avg_dict[key] = (float(val1) + float(val2)) / 2.0
                except (TypeError, ValueError):
                    continue
        return avg_dict

    def sum_absolute_diff(self, dict1, dict2):
        total_diff = 0.0
        for key, val1 in dict1.items():
            if key in dict2:
                val2 = dict2[key]
                try:
                    total_diff += abs(float(val1) - float(val2))
                except (TypeError, ValueError):
                    continue
        return total_diff

    def glcm_texture(self, timage, tmask, settings):
        texFeatures = glcm.RadiomicsGLCM(timage, tmask, **settings)
        texFeatures.enableFeatureByName("Contrast", True)
        texFeatures.enableFeatureByName("Correlation", True)
        texFeatures.enableFeatureByName("JointEnergy", True)
        texFeatures.enableFeatureByName("Autocorrelation", True)
        texFeatures.enableFeatureByName("JointEntropy", True)
        texFeatures.enableFeatureByName("Idm", True)
        texFeatures.enableFeatureByName("ClusterShade", True)
        texFeatures.enableFeatureByName("ClusterTendency", True)
        return texFeatures.execute()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([
            ("G_GAN", self.loss_G_GAN.item()),
            ("G_L1", self.loss_G_L1.item()),
            ("G_MS_SSIM", self.loss_G_MS_SSIM.item()),
            ("G_FM", self.loss_G_FM.item()),
            ("G_GLCM", float(self.glcm_loss)),
            ("D_real", self.loss_D_real.item()),
            ("D_fake", self.loss_D_fake.item()),
        ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([("real_A", real_A), ("fake_B", fake_B), ("real_B", real_B)])

    def save(self, label):
        self.save_network(self.netG, "G", label, self.gpu_ids)
        self.save_network(self.netD, "D", label, self.gpu_ids)

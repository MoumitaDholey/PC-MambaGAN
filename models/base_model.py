import os
import torch

# Modules of the original generator that were constructed but never used in forward():
# the ViT encoder of the base codebase and the alternative fusion blocks of each PC-Mamba block.
LEGACY_UNUSED_PREFIXES = ('transformer_encoder.',)
LEGACY_UNUSED_SUBMODULES = ('gweight', 'eca', 'mifax')


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = torch.load(save_path, map_location='cpu')
        # Checkpoints written by the original code also hold weights of modules that were
        # built but never used in the forward pass. Drop exactly those; any other mismatch
        # still raises an error in load_state_dict.
        unused = [k for k in state_dict if k.startswith(LEGACY_UNUSED_PREFIXES)
                  or any(part in LEGACY_UNUSED_SUBMODULES for part in k.split('.'))]
        if unused:
            print('load_network: ignoring %d unused legacy tensors in %s' % (len(unused), save_filename))
            for k in unused:
                del state_dict[k]
        network.load_state_dict(state_dict)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

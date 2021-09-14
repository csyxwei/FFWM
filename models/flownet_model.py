import torch
from .base_model import BaseModel
from . import base_networks, losses

"""
FlowNet Model
"""


class FlowNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['loss', 'loss_reg', 'loss_lm', 'loss_cor']

        self.flowNet = base_networks.FlowNet(64).to(self.device)
        self.warpNet = base_networks.WarpNet().to(self.device)

        if self.isTrain:
            self.model_names = ['flowNet']
        else:
            self.model_names = ['flowNet']

        if self.isTrain:
            self.criterionLD = losses.MultiScaleLDLoss().to(self.device)
            self.Correctness = losses.PerceptualCorrectness().to(self.device)
            self.Regularization = losses.MultiAffineRegularizationLoss(kz_dic={1: 7, 2: 5, 3: 3}).to(self.device)

            self.optimizer = torch.optim.Adam(self.flowNet.parameters(), lr=0.0004, betas=(0.5, 0.999))

            self.optimizers = [self.optimizer]

    ############# train part

    def set_train_input(self, input):
        self.image_paths = input['input_path']
        if self.reverse:
            self.img_S = input['img_F'].to(self.device).float()
            self.img_F = input['img_S'].to(self.device).float()
            self.lm_S = input['lm_F'].to(self.device).long()
            self.lm_F = input['lm_S'].to(self.device).long()
            self.mask = input['mask_S'].to(self.device).float()
        else:
            self.img_S = input['img_S'].to(self.device).float()
            self.img_F = input['img_F'].to(self.device).float()
            self.lm_S = input['lm_S'].to(self.device).long()
            self.lm_F = input['lm_F'].to(self.device).long()
            self.mask = input['mask_F'].to(self.device).float()
        gate = input['gate'].to(self.device).float()
        self.gate = torch.cat((gate, gate), 2)


    def forward(self):
        if self.reverse:
            self.flow, self.flow64, self.flow32 = self.flowNet(self.img_F) # F is the profile image
        else:
            self.flow, self.flow64, self.flow32 = self.flowNet(self.img_S)
        self.fake_F = self.warpNet(self.img_S, self.flow)

    def backward(self):
        self.flows = [self.flow, self.flow64, self.flow32]

        self.loss_cor = self.Correctness(self.img_F, self.img_S, self.flows[::-1], [2, 1, 0], norm_mask=self.mask) * 20
        self.loss_reg = self.Regularization(self.flows[::-1]) * 0.01
        self.loss_lm = self.criterionLD(self.flows, self.lm_S, self.lm_F, self.gate)

        self.loss = self.loss_cor + self.loss_lm + self.loss_reg
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.zero(self.optimizers)
        self.backward()
        self.step(self.optimizers)

    ############ other part

    def set_visual_name(self):
        if self.isTrain:
            self.visual_names = ['img_S', 'img_F', 'fake_F', 'mask']
        else:
            self.visual_names = ['img_S', 'img_F', 'fake_F']

    def load_network(self, net, path):
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)

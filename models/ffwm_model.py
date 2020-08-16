import torch
from .base_model import BaseModel
from . import losses, external_function, base_networks
import torch.nn.functional as F
from lightcnn.light_cnn import LightCNN_29Layers, LightCNN_29Layers_v2
import itertools

"""
The Flow-based Feature Warping Model
"""

class FFWMModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['loss_G', 'loss_D', 'loss_l1', 'loss_iden', 'loss_illu', 'loss_adv', 'loss_prc', 'loss_fc']

        self.flowNetF = base_networks.FlowNet(64).cuda()
        self.flowNetB = base_networks.FlowNet(64).cuda()
        self.warpNet = base_networks.WarpNet().cuda().eval()
        self.lightCNN = LightCNN_29Layers().cuda().eval()

        self.netG = base_networks.FFWM(sn=True).cuda()
        self.netD = base_networks.MSDiscriminator(128, sigmoid=False).cuda()

        self.load_network(self.lightCNN, opt.lightcnn)

        if self.isTrain:
            self.model_names = ['netG', 'netD', 'flowNetF', 'flowNetB']
            self.load_network(self.flowNetF, opt.flownetf)
            self.load_network(self.flowNetB, opt.flownetb)
        else:
            self.model_names = ['netG', 'flowNetF']

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss().cuda()
            self.criterionIllu = losses.MSL1Loss(self.criterionL1).cuda()
            self.criterionPerceptual = losses.PerceptualLoss().cuda()
            self.criterionIden = losses.IdentityLoss(self.lightCNN, crop=opt.crop).cuda()
            self.criterionGAN = losses.GANLoss('lsgan').cuda()

            self.optimizer_F = torch.optim.Adam(itertools.chain(self.flowNetF.parameters(), self.flowNetB.parameters()),
                                                lr=0.00005, betas=(0.5, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0004, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0004, betas=(0.5, 0.999))

            self.optimizers = [self.optimizer_G, self.optimizer_F, self.optimizer_D]

            self.optimizers_G = [self.optimizer_G, self.optimizer_F]
            self.optimizers_D = [self.optimizer_D]

        # Guided Filter
        self.gf128 = external_function.GuidedFilter(32).cuda()
        self.gf64 = external_function.GuidedFilter(16).cuda()
        self.gf32 = external_function.GuidedFilter(8).cuda()

    def set_train_input(self, input):
        # S is profile, and F is Frontal
        self.image_paths = input['input_path']
        self.img_S = input['img_S'].cuda()
        self.img_F = input['img_F'].cuda()
        self.lm_F = input['lm_F'].cuda()
        self.mask_F = input['mask_F'].cuda().float()
        self.mask_S = input['mask_S'].cuda().float()
        self.titers = input['titers']
        self.epoch = input['epoch']

    def forward(self):
        flow_F128, flow_F64, flow_F32 = self.flowNetF(self.img_S)
        self.img_S_warp = self.warpNet(self.img_S, flow_F128)
        self.flow_B128, self.flow_B64, self.flow_B32 = self.flowNetB(self.img_S)

        self.img_S_rec = self.warpNet(self.img_F, self.flow_B128)

        self.fake_F32, self.fake_F64, self.fake_F128 = self.netG(self.img_S, flow=[flow_F32, flow_F64, flow_F128])

        self.img_GF128 = self.gf128(self.fake_F128, self.img_F)

        # get facial local part, eyes, nose, and mouth
        grid_el, grid_er, grid_n, grid_m = self.get_part_grid()
        self.eyerg, self.eyergt = self.warpNet(self.img_GF128, grid_er), self.warpNet(self.img_F, grid_er)
        self.eyelg, self.eyelgt = self.warpNet(self.img_GF128, grid_el), self.warpNet(self.img_F, grid_el)
        self.noseg, self.nosegt = self.warpNet(self.img_GF128, grid_n), self.warpNet(self.img_F, grid_n)
        self.mouthg, self.mouthgt = self.warpNet(self.img_GF128, grid_m), self.warpNet(self.img_F, grid_m)


    def backward_G(self):
        img_F64 = F.interpolate(self.img_F, (64, 64), mode='bilinear')
        img_F32 = F.interpolate(self.img_F, (32, 32), mode='bilinear')
        mask_F64 = F.interpolate(self.mask_F, (64, 64), mode='nearest')
        mask_F32 = F.interpolate(self.mask_F, (32, 32), mode='nearest')

        if self.titers < 20000:
            #### init model!!! very important!!!
            img_GF128 = self.fake_F128
            img_GF64 = self.fake_F64
            img_GF32 = self.fake_F32
        else:
            img_GF128 = self.img_GF128
            img_GF64 = self.gf64(self.fake_F64, img_F64)
            img_GF32 = self.gf32(self.fake_F32, img_F32)

        loss_prc128 = self.criterionPerceptual(img_GF128 * self.mask_F, self.img_F * self.mask_F)
        loss_prc64 = self.criterionPerceptual(img_GF64 * mask_F64, img_F64 * mask_F64)
        loss_prc32 = self.criterionPerceptual(img_GF32 * mask_F32, img_F32 * mask_F32)
        self.loss_prc = 1 * loss_prc128 + 1 * loss_prc64 + 1.5 * loss_prc32

        loss_l1128 = self.criterionL1(img_GF128 * self.mask_F, self.img_F * self.mask_F)
        loss_l164 = self.criterionL1(img_GF64 * mask_F64, img_F64 * mask_F64)
        loss_l132 = self.criterionL1(img_GF32 * mask_F32, img_F32 * mask_F32)
        self.loss_l1 = 1 * loss_l1128 + 1 * loss_l164 + 1.5 * loss_l132

        self.loss_illu = self.criterionIllu([self.flow_B128, self.flow_B64, self.flow_B32],
                                            [self.fake_F128, self.fake_F64, self.fake_F32],
                                            self.img_S, self.mask_S)

        self.loss_iden = self.criterionIden(self.fake_F128, self.img_F)
        self.loss_iden_gf = self.criterionIden(img_GF128, self.img_F)

        gen_fake = self.netD(self.img_GF128 * self.mask_F)
        self.loss_adv = self.criterionGAN(gen_fake, True, for_dis=False)

        loss_prc_fc_e = self.criterionPerceptual(self.eyelg, self.eyelgt) + \
            self.criterionPerceptual(self.eyerg, self.eyergt)
        loss_prc_fc_n = self.criterionPerceptual(self.noseg, self.nosegt)
        loss_prc_fc_m = self.criterionPerceptual(self.mouthg, self.mouthgt)
        self.loss_fc = 2 * loss_prc_fc_e + loss_prc_fc_m + loss_prc_fc_n

        self.loss_prc = self.loss_prc * 1
        self.loss_fc = self.loss_fc * 1
        self.loss_l1 = self.loss_l1 * 5
        self.loss_iden = self.loss_iden * 0.5 + self.loss_iden_gf * 1
        self.loss_adv = self.loss_adv * 0.1
        self.loss_illu = self.loss_illu * 15
        self.loss_G = self.loss_iden + self.loss_l1 + self.loss_prc + self.loss_illu + self.loss_fc  + self.loss_adv
        self.loss_G.backward()


    def backward_D(self):
        dis_fake = self.netD(self.img_GF128.detach() * self.mask_F)
        dis_real = self.netD(self.img_F * self.mask_F)
        loss_D_fake = self.criterionGAN(dis_fake, False, for_dis=True)
        loss_D_real = self.criterionGAN(dis_real, True, for_dis=True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.zero(self.optimizers_D)
        self.backward_D()
        self.step(self.optimizers_D)
        self.set_requires_grad(self.netD, False)
        self.zero(self.optimizers_G)
        self.backward_G()
        self.step(self.optimizers_G)

    ############# test part

    def get_gallery_fea(self, keys, gallery):
        feas = []
        for key in keys:
            tensor = gallery[key].cuda()
            if len(tensor.size()) == 3:
                tensor = tensor.unsqueeze(0)
            if self.opt.crop:
                grid = self.criterionIden.build_grid(tensor.size(0), 98)
                tensor = self.warpNet(tensor, grid)
                tensor = F.interpolate(tensor, (128, 128), mode='bilinear')
            _, f, _ = self.lightCNN(tensor)
            feas.append(f.detach())
        return torch.cat(feas, 0)

    def set_test_input(self, input):
        self.image_paths = input['input_path']
        self.img_S = input['img_S'].cuda()
        self.img_F = input['img_F'].cuda()

    def test_forward(self):
        flow_F128, flow_F64, flow_F32 = self.flowNetF(self.img_S)
        self.flow = flow_F128
        self.img_S_warp = self.warpNet(self.img_S, flow_F128)
        _, _, self.fake_F128, att = self.netG(self.img_S, flow=[flow_F32, flow_F64, flow_F128], return_att=True)
        self.att = torch.mean(att[:, :64, :, :], (1, ), keepdim=True)
        self.img_GF128 = self.gf128(self.fake_F128, self.img_F)

    def test(self, return_fea=True):
        with torch.no_grad():
            self.test_forward()
            self.compute_visuals()
            if return_fea:
                fake_F_gray = torch.mean(self.fake_F128, dim=(1,), keepdim=True)
                if self.opt.crop:
                    grid = self.criterionIden.build_grid(self.fake_F128.size(0), 98)
                    fake_F_gray = self.warpNet(fake_F_gray, grid)
                    fake_F_gray = F.interpolate(fake_F_gray, (128, 128), mode='bilinear')
                _, fea, _ = self.lightCNN(fake_F_gray)
                return fea.detach()

    ############### other part

    def set_visual_name(self):
        if self.isTrain:
            self.visual_names = ['img_S', 'img_F', 'img_S_warp', 'fake_F32', 'fake_F64', 'fake_F128', 'img_S_rec', 'img_GF128']
        else:
            self.visual_names = ['img_S', 'img_F', 'fake_F128']

    def load_network(self, net, path):
        print('loading the model from ', path)
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)

    def get_part_grid(self):
        """
        get the facial part grids which can be used to crop facial part from the original face image.
        """
        # the facial part center landmarks, modify according to your data
        el, er = self.lm_F[:, 63:64], self.lm_F[:, 515:516]
        ml, mr = self.lm_F[:, 64:128], self.lm_F[:, 516:580]
        nc = self.lm_F[:, 429:430]
        mc = torch.cat((ml, mr), 1)
        mc = torch.min(mc, dim=1, keepdim=True)[0] + torch.max(mc, dim=1, keepdim=True)[0]
        mc = mc / 2
        grid_el = self.build_grid(el, 32)
        grid_er = self.build_grid(er, 32)
        grid_n = self.build_grid(nc, 32)
        grid_m = self.build_grid(mc, 32)
        return grid_el, grid_er, grid_n, grid_m

    def build_grid(self, lm, d):
        """
        build a grid to crop local patch based the facial landmark (lm). d is the patch diameter.
        """
        b = lm.size(0)
        r = d // 2
        base_x = torch.linspace(-r, r, d).cuda().unsqueeze(0).repeat(d, 1).unsqueeze(-1)
        base = torch.cat([base_x, base_x.transpose(1, 0)], dim=2).unsqueeze(0)
        base = base.repeat(b, 1, 1 ,1)
        bias = lm.unsqueeze(1).float()
        bias = bias.repeat(1, d, d, 1) - 64 # from [0, 127] to [-64, 63]
        grid = (base + bias) / 64 # / 64 is scale to [-1, 1]
        return grid.transpose(2, 3).transpose(1, 2)

    def load_pretrain(self, prefix, epoch):
        if not prefix.endswith('/'):
            prefix = prefix + '/'
        for m in self.model_names:
            self.load_network(getattr(self, m), prefix + '{}_net_{}.pth'.format(epoch, m))
            print('loaded ' + m)
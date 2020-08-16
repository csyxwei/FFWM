import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'nsgan':
            self.criterion = nn.BCELoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.criterion = nn.ReLU()
        elif gan_mode in ['wgangp', 'dcgan']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real, for_dis=None, weights=None):
        loss = 0
        if type(predictions) is not list:
            predictions = [predictions]
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla', 'nagan']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss += self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss += -prediction.mean()
                else:
                    loss += prediction.mean()
            elif self.gan_mode == 'hinge':
                if for_dis:
                    if target_is_real:
                        prediction = - prediction
                    loss += self.criterion(1 + prediction).mean()
                else:
                    loss += (-prediction).mean()
            elif self.gan_mode == 'dcgan':
                if target_is_real:
                    loss += torch.mean(F.softplus(-prediction))
                else:
                    loss += torch.mean(F.softplus(prediction))
        return loss

class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.criterionL2 = torch.nn.MSELoss()

    def forward(self, flow, lm_S, lm_F, gate):
        b, _, s, _ = flow.size()
        flow_view = flow.transpose(1, 2).transpose(2, 3).view(b, -1, 2)
        index = lm_F[:, :, 0:1] + lm_F[:, :, 1:2] * s
        index = torch.cat((index, index), 2)
        flow_points = torch.gather(flow_view, 1, index)
        gt_points = lm_S.float() / (s / 2.0) - 1

        return self.criterionL2(flow_points * gate, gt_points * gate)

class IdentityLoss(nn.Module):
    def __init__(self, lightcnn, crop=False):
        super(IdentityLoss, self).__init__()
        self.lightcnn = lightcnn
        self.criterionL1 = torch.nn.L1Loss()
        self.warpNet = base_networks.WarpNet()
        self.crop = crop

    def forward(self, out, gt):
        grid = self.build_grid(out.size(0), 98)
        # crop the center face image
        if self.crop:
            input_out, input_gt = self.warpNet(out, grid), self.warpNet(gt, grid)
            out = F.interpolate(input_out, (out.size(2), out.size(3)), mode='bilinear')
            gt = F.interpolate(input_gt, (out.size(2), out.size(3)), mode='bilinear')
        input_out = torch.mean(out, dim=(1,), keepdim=True)
        _, fc_out, pool_out = self.lightcnn(input_out)
        input_gt = torch.mean(gt, dim=(1,), keepdim=True)

        with torch.no_grad():
            _, fc_gt, pool_gt = self.lightcnn(input_gt)

        loss = self.criterionL1(fc_out, fc_gt.detach())
        loss += self.criterionL1(pool_out, pool_gt.detach())
        return loss

    def build_grid(self, b, d):
        r = d // 2
        base_x = torch.linspace(-r, r, d).cuda().unsqueeze(0).repeat(d, 1).unsqueeze(-1)
        base = torch.cat([base_x, base_x.transpose(1, 0)], dim=2).unsqueeze(0)
        base = base.repeat(b, 1, 1 ,1)
        bias = torch.zeros_like(base)
        bias[:, :, :, 0] = 64
        bias[:, :, :, 1] = 77
        bias = bias - 64
        grid = (base + bias) / 64
        return grid.transpose(2, 3).transpose(1, 2)

class MultiScaleLDLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLDLoss, self).__init__()
        self.criterionLD = LandmarkLoss()
        self.weights = [1000, 1000, 1500]
        self.img_size = 128

    def forward(self, flows, lm_S, lm_F, gate):
        ld_loss = 0
        for i, flow in enumerate(flows):
            scale = self.img_size // flow.size(3)
            ld_loss += self.weights[i] * self.criterionLD(flow, lm_S.div(scale), lm_F.div(scale), gate)
        return ld_loss

from . import base_networks

class MSL1Loss(nn.Module):
    def __init__(self, criterionL1):
        super(MSL1Loss, self).__init__()
        self.warpNet = base_networks.WarpNet().cuda()
        self.criterionL1 = criterionL1
        self.l1_weights = [1, 1, 1.5]

    def resize_as(self, img, tar, mode='bilinear'):
        b, _, h, w = tar.size()
        if mode == 'bilinear':
            return F.interpolate(img, (h, w), mode=mode, align_corners=True)
        else:
            return F.interpolate(img, (h, w), mode=mode)

    def forward(self, flows, img_Ss, img_F, mask=None):
        loss_l1 = 0
        for i, flow in enumerate(flows):
            _img_S = img_Ss[i]
            _img_F = self.resize_as(img_F, flow)
            _fake_F = self.warpNet(_img_S, flow)
            if mask is None:
                loss_l1 += self.l1_weights[i] * self.criterionL1(_fake_F, _img_F)
            else:
                _mask = self.resize_as(mask, flow, 'nearest')
                loss_l1 += self.l1_weights[i] * self.criterionL1(_fake_F * _mask, _img_F * _mask)
        return loss_l1



##### Losses borrowed from https://github.com/RenYurui/Global-Flow-Local-Attention/blob/master/model/networks/external_function.py

from .external_function import Resample2d, LocalAttnReshape, BlockExtractor

class MultiAffineRegularizationLoss(nn.Module):
    def __init__(self, kz_dic):
        super(MultiAffineRegularizationLoss, self).__init__()
        self.kz_dic = kz_dic
        self.method_dic = {}
        for key in kz_dic:
            instance = AffineRegularizationLoss(kz_dic[key])
            self.method_dic[key] = instance
        self.layers = sorted(kz_dic, reverse=True)

    def __call__(self, flow_fields):
        loss = 0
        for i in range(len(flow_fields)):
            method = self.method_dic[self.layers[i]]
            loss += method(flow_fields[i])
        return loss


class AffineRegularizationLoss(nn.Module):
    """docstring for AffineRegularizationLoss"""

    # kernel_size: kz
    def __init__(self, kz):
        super(AffineRegularizationLoss, self).__init__()
        self.kz = kz
        self.criterion = torch.nn.L1Loss()
        self.extractor = BlockExtractor(kernel_size=kz)
        self.reshape = LocalAttnReshape()

        temp = np.arange(kz)
        A = np.ones([kz * kz, 3])
        A[:, 0] = temp.repeat(kz)
        A[:, 1] = temp.repeat(kz).reshape((kz, kz)).transpose().reshape(kz ** 2)
        AH = A.transpose()
        k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz ** 2)  # K = (A((AH A)^-1)AH - I)
        self.kernel = np.dot(k.transpose(), k)
        self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz ** 2, kz, kz).unsqueeze(1)

    def __call__(self, flow_fields):
        grid = self.flow2grid(flow_fields)

        grid_x = grid[:, 0, :, :].unsqueeze(1)
        grid_y = grid[:, 1, :, :].unsqueeze(1)
        weights = self.kernel.type_as(flow_fields)
        loss_x = self.calculate_loss(grid_x, weights)
        loss_y = self.calculate_loss(grid_y, weights)
        return loss_x + loss_y

    def calculate_loss(self, grid, weights):
        results = nn.functional.conv2d(grid, weights)  # KH K B [b, kz*kz, w, h]
        b, c, h, w = results.size()
        kernels_new = self.reshape(results, self.kz)
        f = torch.zeros(b, 2, h, w).type_as(kernels_new) + float(int(self.kz / 2))
        grid_H = self.extractor(grid, f)
        result = torch.nn.functional.avg_pool2d(grid_H * kernels_new, self.kz, self.kz)
        loss = torch.mean(result) * self.kz ** 2
        return loss

    def flow2grid(self, flow_field):
        grid = flow_field.add(1.0).div(2.0).mul(128.0)
        return grid

class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return content_loss, style_loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    #  layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    def __init__(self, layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                 weights=[1.0, 1.0 / 2, 1.0 / 4, 1.0 / 4, 1.0 / 8]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.layers = layers

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            content_loss += weight * self.criterion(x_vgg[layer], y_vgg[layer].detach())

        return content_loss


class PerceptualCorrectness(nn.Module):

    def __init__(self, layer=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps = 1e-8
        self.resample = Resample2d(4, 1, sigma=2)
        self.l1_loss = nn.L1Loss()

    def __call__(self, target, source, flow_list, used_layers, norm_mask=None, use_bilinear_sampling=True):
        used_layers = sorted(used_layers, reverse=True)
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], norm_mask, use_bilinear_sampling)

        return loss

    def calculate_loss(self, flow, layer, norm_mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        flow = F.interpolate(flow, [h, w])

        target_all = target_vgg.view(b, c, -1)  # [b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1, 2)  # [b N2 C]

        source_norm = source_all / (source_all.norm(dim=2, keepdim=True) + self.eps)
        target_norm = target_all / (target_all.norm(dim=1, keepdim=True) + self.eps)
        correction = torch.bmm(source_norm, target_norm)  # [b N2 N2]
        (correction_max, max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)  # [b 1 N2]
        loss_map = torch.exp(-correction_sample / (correction_max + self.eps))
        if norm_mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            norm_mask = F.interpolate(norm_mask, size=(target_vgg.size(2), target_vgg.size(3)))
            norm_mask = norm_mask.view(-1, target_vgg.size(2) * target_vgg.size(3))
            loss = (torch.sum(norm_mask * loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))) / (
                        torch.sum(norm_mask) + self.eps)

        return loss

    def perceptual_loss(self, flow, layer, norm_mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        flow = F.interpolate(flow, [h, w])

        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        if norm_mask is None:
            loss = self.l1_loss(input_sample, target_vgg)
        else:
            norm_mask = F.interpolate(norm_mask, size=(target_vgg.size(2), target_vgg.size(3)))
            loss = self.l1_loss(input_sample * norm_mask, target_vgg * norm_mask)

        return loss

    def bilinear_warp(self, source, flow, view=True):
        b, c = source.shape[:2]
        grid = flow.permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid, mode='bilinear')
        return input_sample

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm
import numpy as np

def initialize_msra(modules):
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


################################################################################################
#### FlowNet mainly borrowed from https://github.com/NVIDIA/flownet2-pytorch
################################################################################################
def conv(in_planes, out_planes, norm_layer=nn.BatchNorm2d, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                  bias=True),
        norm_layer(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )

def deconv(in_planes, out_planes, norm_layer):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        norm_layer(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )

def predict_flow(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Tanh(),
    )

def i_conv(in_planes, out_planes, norm_layer, kernel_size=3, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                  bias=bias),
        norm_layer(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )

class FlowNet(nn.Module):
    def __init__(self, ngf, norm=nn.BatchNorm2d, x=3):
        super(FlowNet, self).__init__()

        self.batchNorm = norm
        self.conv0 = conv(x, ngf * 1, self.batchNorm)
        self.conv1 = conv(ngf * 1, ngf * 1, self.batchNorm, stride=2)
        self.conv1_1 = conv(ngf * 1, ngf * 2, self.batchNorm)
        self.conv2 = conv(ngf * 2, ngf * 2, self.batchNorm, stride=2)
        self.conv2_1 = conv(ngf * 2, ngf * 2, self.batchNorm)
        self.conv3 = conv(ngf * 2, ngf * 4, self.batchNorm, stride=2)
        self.conv3_1 = conv(ngf * 4, ngf * 4, self.batchNorm)
        self.conv4 = conv(ngf * 4, ngf * 8, self.batchNorm, stride=2)
        self.conv4_1 = conv(ngf * 8, ngf * 8, self.batchNorm)
        self.conv5 = conv(ngf * 8, ngf * 8, self.batchNorm, stride=2)
        self.conv5_1 = conv(ngf * 8, ngf * 8, self.batchNorm)
        self.conv6 = conv(ngf * 8, ngf * 16, self.batchNorm, stride=2)
        self.conv6_1 = conv(ngf * 16, ngf * 16, self.batchNorm)

        self.deconv5 = deconv(ngf * 16, ngf * 8, self.batchNorm)
        self.deconv4 = deconv(ngf * 16 + 2, ngf * 4, self.batchNorm)
        self.deconv3 = deconv(ngf * 8 + ngf * 4 + 2, ngf * 2, self.batchNorm)
        self.deconv2 = deconv(ngf * 4 + ngf * 2 + 2, ngf * 1, self.batchNorm)
        self.deconv1 = deconv(ngf * 1 + 2, ngf // 2, self.batchNorm)
        self.deconv0 = deconv(ngf // 2 + 2, ngf // 4, self.batchNorm)

        self.inter_conv5 = i_conv(ngf * 16 + 2, ngf * 8, self.batchNorm)
        self.inter_conv4 = i_conv(ngf * 8 + ngf * 4 + 2, ngf * 4, self.batchNorm)
        self.inter_conv3 = i_conv(ngf * 4 + ngf * 2 + 2, ngf * 2, self.batchNorm)
        self.inter_conv2 = i_conv(ngf * 1 + 2, ngf * 1, self.batchNorm)
        self.inter_conv1 = i_conv(ngf // 2 + 2, ngf // 2, self.batchNorm)
        self.inter_conv0 = i_conv(ngf // 4 + 2, ngf // 4, self.batchNorm)

        self.inter_conv_occ5 = i_conv(ngf * 16 + 1, ngf * 8, self.batchNorm)
        self.inter_conv_occ4 = i_conv(ngf * 8 + ngf * 4 + 1, ngf * 4, self.batchNorm)
        self.inter_conv_occ3 = i_conv(ngf * 4 + ngf * 2 + 1, ngf * 2, self.batchNorm)
        self.inter_conv_occ2 = i_conv(ngf * 1 + 1, ngf * 1, self.batchNorm)
        self.inter_conv_occ1 = i_conv(ngf // 2 + 1, ngf // 2, self.batchNorm)
        self.inter_conv_occ0 = i_conv(ngf // 4 + 1, ngf // 4, self.batchNorm)

        self.predict_flow6 = predict_flow(ngf * 16)
        self.predict_flow5 = predict_flow(ngf * 8)
        self.predict_flow4 = predict_flow(ngf * 4)
        self.predict_flow3 = predict_flow(ngf * 2)
        self.predict_flow2 = predict_flow(ngf * 1)
        self.predict_flow1 = predict_flow(ngf // 2)
        self.predict_flow0 = predict_flow(ngf // 4)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        initialize_msra(self.modules())

    def forward(self, x):
        # inputxy = X
        out_conv0 = self.conv0(x)  # output: B*(ngf)*128*128
        out_conv1 = self.conv1_1(self.conv1(out_conv0))  # output: B*(ngf*2)*64*64
        out_conv2 = self.conv2_1(self.conv2(out_conv1))  # output: B*(ngf*2)*32*32
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # output: B*(ngf*4)*16*16
        out_conv4 = self.conv4_1(self.conv4(out_conv3))  # output: B*(ngf*8)*8*8
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # output: B*(ngf*8)*4*4
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # output: B*(ngf*16)*2*2

        # Flow Decoder
        flow6 = self.predict_flow6(out_conv6)  # output: B*2*2*2

        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)  # output: B*2*4*4

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)  # output: B*2*8*8

        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)  # output: B*2*16*16

        flow3_up = self.upsampled_flow3_to_2(flow3)  # output: B*2*32*32
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)  # output: B*2*32*32

        flow2_up = self.upsampled_flow2_to_1(flow2)  # output: B*2*64*64
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)  # output: B*2*64*64

        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)  # output: B*2*128*128

        return flow0, flow1, flow2


class WarpNet(nn.Module):
    def __init__(self):
        super(WarpNet, self).__init__()

    def forward(self, images, flow, mode='bilinear'):
        return F.grid_sample(images, flow.transpose(1, 2).transpose(2, 3), mode=mode)

################################################################################################
#### Generator
################################################################################################

class Tanh2(nn.Module):
    def __init__(self):
        super(Tanh2, self).__init__()
        self.tanh = nn.Tanh()
    def forward(self, x):
        return (self.tanh(x) + 1) / 2

def get_activ(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'tanh2':
        return Tanh2()
    else:
        raise NotImplementedError('Activation %s not implemented' % name)

def get_norm(name, ch):
    if name == 'bn':
        return nn.BatchNorm2d(ch)
    elif name == 'in':
        return nn.InstanceNorm2d(ch)
    else:
        raise NotImplementedError('Normalization %s not implemented' % name)

class ResidualBlock(nn.Module):
    def __init__(self, inc, outc=None, kernel=3, stride=1, activ='lrelu', norm='bn', sn=False):
        super(ResidualBlock, self).__init__()

        if outc is None:
            outc = inc // stride

        self.activ = get_activ(activ)
        pad = kernel // 2
        if sn:
            self.input = SpectralNorm(nn.Conv2d(inc, outc, 1, 1, padding=0))
            self.blocks = nn.Sequential(SpectralNorm(nn.Conv2d(inc, outc, kernel, 1, pad)),
                                        get_norm(norm, outc),
                                        nn.LeakyReLU(0.2),
                                        SpectralNorm(nn.Conv2d(outc, outc, kernel, 1, pad)),
                                        get_norm(norm, outc))
        else:
            self.input = nn.Conv2d(inc, outc, 1, 1, padding=0)
            self.blocks = nn.Sequential(nn.Conv2d(inc, outc, kernel, 1, kernel),
                                        get_norm(norm, outc),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(outc, outc, kernel, 1, kernel),
                                        get_norm(norm, outc))

    def forward(self, x):
        return self.activ(self.blocks(x) + self.input(x))

def ConvBlock(inc, outc, ks=3, s=1, p=0, activ='lrelu', norm='bn', res=0, resk=3, bn=True, sn=False):
    conv = nn.Conv2d(inc, outc, ks, s, p)
    if sn:
        conv = SpectralNorm(conv)
    blocks = [conv]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, kernel=resk, norm=norm, sn=sn))
    return nn.Sequential(*blocks)

def DeConvBlock(inc, outc, ks=3, s=1, p=0, op=0, activ='relu', norm='bn', res=0, resk=3, bn=True, sn=False):
    deconv = nn.ConvTranspose2d(inc, outc, ks, s, p, op)
    if sn:
        deconv = SpectralNorm(deconv)
    blocks = [deconv]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, norm=norm, kernel=resk, sn=sn))
    return nn.Sequential(*blocks)

def PixelSuffleBlock(inc, outc, ks=3, s=1, p=0, activ='lrelu', norm='bn', res=0, bn=True, sn=False):
    conv = nn.Conv2d(inc, outc * 4, 3, 1, 1)
    if sn:
        conv = SpectralNorm(conv)
    blocks = [conv, nn.PixelShuffle(2)]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, norm=norm, sn=sn))
    return nn.Sequential(*blocks)

class FFWM(nn.Module):
    def __init__(self, num_layers=3, isflip=True, sn=False):
        super(FFWM, self).__init__()
        channels = [64, 64, 128, 256]
        dechannels = [256, 128, 64, 64]
        self.isflip = isflip
        dm = 3 if isflip else 2
        am = dm - 1

        self.layers = num_layers
        self.e0 = ConvBlock(3, channels[0], 7, 1, 3, res=1, bn=False, sn=sn)
        self.e1 = ConvBlock(channels[0], channels[1], 4, 2, 1, res=1, sn=sn)
        self.e2 = ConvBlock(channels[1], channels[2], 4, 2, 1, res=1, sn=sn)
        self.e3 = ConvBlock(channels[2], channels[3], 4, 2, 1, res=1, sn=sn)

        self.d0 = PixelSuffleBlock(dechannels[0], dechannels[1], 4, 2, 1, sn=sn)
        self.d1 = PixelSuffleBlock(dechannels[1] * dm, dechannels[2], 4, 2, 1, sn=sn)
        self.d2 = PixelSuffleBlock(dechannels[2] * dm + 3, dechannels[3], 4, 2, 1, sn=sn)

        self.dres0 = nn.Sequential(*[ResidualBlock(dechannels[1] * dm, activ='lrelu', sn=sn)
                                    for i in range(2)])
        self.dres1 = nn.Sequential(*[ResidualBlock(dechannels[2] * dm + 3, activ='lrelu', sn=sn)
                                    for i in range(2)])
        self.dres2 = nn.Sequential(*[ResidualBlock(dechannels[3] * dm + 3, activ='lrelu', sn=sn)
                                    for i in range(2)])

        self.rec0 = ConvBlock(dechannels[1] * dm, 3, 3, 1, 1, bn=False, activ='sigmoid', sn=sn)
        self.rec1 = ConvBlock(dechannels[2] * dm + 3, 3, 3, 1, 1, bn=False, activ='sigmoid', sn=sn)
        self.rec2 = ConvBlock(dechannels[3] * dm + 3, 3, 3, 1, 1, bn=False, activ='sigmoid', sn=sn)

        self.att0 = nn.Sequential(ConvBlock(channels[2] * am, channels[2] * am, 3, 1, 1, sn=sn),
                                  ResidualBlock(channels[2] * am, channels[2] * am, activ='sigmoid', sn=sn))
        self.att1 = nn.Sequential(ConvBlock(channels[1] * am, channels[1] * am, 3, 1, 1, sn=sn),
                                  ResidualBlock(channels[1] * am, channels[1] * am, activ='sigmoid', sn=sn))
        self.att2 = nn.Sequential(ConvBlock(channels[0] * am, channels[0] * am, 3, 1, 1, sn=sn),
                                  ResidualBlock(channels[0] * am, channels[0] * am, activ='sigmoid', sn=sn))

        self.warpNet = WarpNet()
        # initialize_msra(self.modules())

    def forward(self, x, flow=None, return_att=False):
        fencs = [self.e0(x)]
        for i in range(1, self.layers + 1):
            fencs.append(getattr(self, 'e{}'.format(i))(fencs[-1]))

        fdec = fencs[-1]
        _fencs = fencs[::-1]
        recons = []
        att = None
        for i in range(self.layers):
            dec = getattr(self, 'd{}'.format(i))(fdec)
            # Warp Attention Module: warp, flip, and attention
            w =  self.warpNet(_fencs[i + 1], flow[i])
            if self.isflip:
                _w = torch.flip(w, (3,))
                skip = torch.cat((w, _w), 1)
            else:
                skip = w
            att = getattr(self, 'att{}'.format(i))(skip)
            skip = skip * att

            # following TP-GAN, add the low resolution reconstructed image to decoder
            if len(recons) > 0:
                res_in = torch.cat((skip, dec, F.interpolate(recons[-1], scale_factor=2, mode='bilinear')), 1)
            else:
                res_in = torch.cat((skip, dec), 1)
            fdec = getattr(self, 'dres{}'.format(i))(res_in)
            # reconstruct
            recons.append(getattr(self, 'rec{}'.format(i))(fdec))

        if return_att: # for visulizing the attention map
            return recons[-3], recons[-2], recons[-1], att
        else:
            return recons[-3], recons[-2], recons[-1]


################################################################################################
#### Discriminator mainly borrowed from https://github.com/assafshocher/InGAN
################################################################################################

class MSDiscriminator(nn.Module):
    def __init__(self, real_crop_size, inc=3, max_n_scales=9, scale_factor=2, base_channels=64, extra_conv_layers=0, sigmoid=True):
        super(MSDiscriminator, self).__init__()
        self.inc = inc
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.min_size = 16
        self.extra_conv_layers = extra_conv_layers
        self.sigmoid = sigmoid

        # We want the max num of scales to fit the size of the real examples. further scaling would create networks that
        # only train on fake examples
        self.max_n_scales = np.min([np.int(np.ceil(np.log(np.min(real_crop_size) * 1.0 / self.min_size)
                                                   / np.log(self.scale_factor))), max_n_scales])

        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()

        # Create a network for each scale
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())

    def make_net(self):
        base_channels = self.base_channels
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(nn.Conv2d(self.inc, base_channels, kernel_size=3, stride=2, padding=1)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                 out_channels=base_channels * 4,
                                                 kernel_size=3,
                                                 stride=2,
                                                 padding=1,
                                                 bias=True)),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                     out_channels=base_channels * 2,
                                                     kernel_size=3,
                                                     bias=True)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]


        # Final conv-block
        if self.sigmoid:
            net += nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(base_channels * 4, 1, kernel_size=1)),
                                 nn.Sigmoid())
        else:
            net += [nn.Conv2d(base_channels * 4, 1, kernel_size=1)]

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor):
        scale_weights = [1, 1, 1, 1, 1]
        aggregated_result_maps_from_all_scales = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Run all nets over all scales and aggregate the interpolated results
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            downscaled_image = F.interpolate(input_tensor, scale_factor=self.scale_factor**(-i), mode='bilinear')
            result_map_for_current_scale = net(downscaled_image)
            upscaled_result_map_for_current_scale = F.interpolate(result_map_for_current_scale,
                                                                  size=map_size,
                                                                  mode='bilinear')
            aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weight

        # aggregated_result_maps_from_all_scales = aggregated_result_maps_from_all_scales.view(-1, 1 * map_size[0] * map_size[1]).view(-1)

        return aggregated_result_maps_from_all_scales

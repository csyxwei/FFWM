import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

try:
    import resample2d_cuda
    import local_attn_reshape_cuda
    import block_extractor_cuda
except ImportError:
    print('Warning! Import resample2d_cuda/local_attn_reshape_cuda/block_extractor_cuda, If you are training network, please install them firstly.')
    print()


#################################################
# borrowed from from https://github.com/RenYurui/Global-Flow-Local-Attention
#################################################

class BlockExtractorFunction(Function):

    @staticmethod
    def forward(ctx, source, flow_field, kernel_size):
        assert source.is_contiguous()
        assert flow_field.is_contiguous()

        # TODO: check the shape of the inputs
        bs, ds, hs, ws = source.size()
        bf, df, hf, wf = flow_field.size()
        # assert bs==bf and hs==hf and ws==wf
        assert df==2

        ctx.save_for_backward(source, flow_field)
        ctx.kernel_size = kernel_size

        output = flow_field.new(bs, ds, kernel_size*hf, kernel_size*wf).zero_()

        if not source.is_cuda:
            raise NotImplementedError
        else:
            block_extractor_cuda.forward(source, flow_field, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        source, flow_field = ctx.saved_tensors
        grad_source = Variable(source.new(source.size()).zero_())
        grad_flow_field = Variable(flow_field.new(flow_field.size()).zero_())

        block_extractor_cuda.backward(source, flow_field, grad_output.data,
                                 grad_source.data, grad_flow_field.data,
                                 ctx.kernel_size)

        return grad_source, grad_flow_field, None

class BlockExtractor(nn.Module):
    def __init__(self, kernel_size=3):
        super(BlockExtractor, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, source, flow_field):
        source_c = source.contiguous()
        flow_field_c = flow_field.contiguous()
        return BlockExtractorFunction.apply(source_c, flow_field_c,
                                          self.kernel_size)

class LocalAttnReshapeFunction(Function):

    @staticmethod
    def forward(ctx, inputs, kernel_size):
        assert inputs.is_contiguous()

        # TODO: check the shape of the inputs
        bs, ds, hs, ws = inputs.size()
        assert ds == kernel_size*kernel_size

        ctx.save_for_backward(inputs)
        ctx.kernel_size = kernel_size

        output = inputs.new(bs, 1, kernel_size*hs, kernel_size*ws).zero_()

        if not inputs.is_cuda:
            raise NotImplementedError
        else:
            local_attn_reshape_cuda.forward(inputs, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        inputs, = ctx.saved_tensors
        grad_inputs = Variable(inputs.new(inputs.size()).zero_())

        local_attn_reshape_cuda.backward(inputs, grad_output.data,
                                 grad_inputs.data, ctx.kernel_size)

        return grad_inputs, None

class LocalAttnReshape(nn.Module):
    def __init__(self):
        super(LocalAttnReshape, self).__init__()

    def forward(self, inputs, kernel_size=3):
        inputs_c = inputs.contiguous()
        return LocalAttnReshapeFunction.apply(inputs_c, kernel_size)

class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=2, dilation=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()

        resample2d_cuda.forward(input1, input2, output, kernel_size, dilation)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size, ctx.dilation)

        return grad_input1, grad_input2, None, None

class Resample2d(nn.Module):

    def __init__(self, kernel_size=2, dilation=1, sigma=5 ):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = torch.tensor(sigma, dtype=torch.float)

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        sigma = self.sigma.expand(input2.size(0), 1, input2.size(2), input2.size(3)).type(input2.dtype)
        input2 = torch.cat((input2,sigma), 1)
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.dilation)

#################################################
# Guided Filter borrowed from https://github.com/wuhuikai/DeepGuidedFilter
#################################################

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b
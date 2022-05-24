import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchsummary import summary
import cv2
import common
import time
from torch.nn import Parameter as P
import math


def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        """
        return pixel_unshuffle(input, self.downscale_factor)

class Pixelshuffle(nn.Module):
    def __init__(self,upscale_factor):
        super(Pixelshuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self,input):
        return F.pixel_shuffle(input, self.upscale_factor)

def build_window(windows):
    kernel = np.zeros(( windows * windows,1, windows, windows), dtype=np.float32)
    for i in range(windows * windows):
        kernel[i, 0, int(i / windows), int(i % windows)] = 1
    kernel = Variable(torch.from_numpy(kernel),requires_grad=False)
    return kernel

class FilterLayer(nn.Module):
    def __init__(self,windows):
        super(FilterLayer, self).__init__()
        self.kernel = build_window(windows).cuda()
        self.zeropadding = nn.ZeroPad2d(windows//2)

    def forward(self, x, f, window):
        x = self.zeropadding(x)
        # print(x.shape) (8, 3, 74, 74)
        _xb = F.conv2d(x[:, 0:1, :, :], self.kernel)
        _xg = F.conv2d(x[:, 1:2, :, :], self.kernel)
        _xr = F.conv2d(x[:, 2:3, :, :], self.kernel)
        yb = torch.sum(_xb * f, dim=1, keepdim=True)
        yg = torch.sum(_xg * f, dim=1, keepdim=True)
        yr = torch.sum(_xr * f, dim=1, keepdim=True)
        y = torch.cat([yb, yg, yr], dim=1)
        # print(y.shape) (8, 3, 64, 64)
        return y

class residual_block(nn.Module):
    def __init__(self, inchannels=128):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=64,kernel_size=3, stride=1, padding= 3 // 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.conv4 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)

    def forward(self, x):
        t1 = F.relu(self.conv1(x))
        t1 = torch.cat((x, t1), 1)
        # print(t1.shape) (8,192,128,128)
        t2 = F.relu(self.conv2(t1))
        t2 = torch.cat((t1,t2),1)
        # print(t2.shape)  (8,256,128,128)
        t3 = F.relu(self.conv3(t2))
        t3 = torch.cat((t2,t3),1)
        # print(t3.shape)    (8,320,128,128)
        t4 = F.relu(self.conv4(t3))
        t4 = torch.cat((t3,t4),1)

        t5 = F.relu(self.conv5(t4))
        t5 = torch.cat((t4,t5),1)
        return t5

class pre_block(nn.Module):
    def __init__(self, inchannels):
        super(pre_block, self).__init__()
        self.conv1 = nn.Conv2d(448, 256, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(448, 128, 1,stride=1, padding=0, bias=True)
        self.residual_block = residual_block(inchannels)

    def forward(self, x):
        t = x
        # print(t.shape) (8,128,128,128)
        t = self.residual_block(t)
        # print(t.shape) (8,448,128,128)
        a = F.relu(self.conv1(t))
        a = F.relu(self.conv2(a))
        a = F.relu(self.conv3(a))
        # print(a.shape)  (8,1,128,128)
        a =torch.sign(a)
        _a = (1-a)
        t = self.conv4(t)
        _t = torch.multiply(t, a)
        # print(_t.shape)  (8,128,128,128)
        _x = torch.multiply(x, _a)
        # print(_x.shape)  (8,128,128,128)
        t = torch.add( _x, _t )
        return t

class pos_block(nn.Module):
    def __init__(self,inchannels):
        super(pos_block,self).__init__()
        self.residual_block = residual_block(inchannels)
        self.conv = nn.Conv2d(448, 128, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        t = x
        t = self.residual_block(t)
        t = F.relu(self.conv(t))
        return t

class filter_block(nn.Module):
    def __init__(self, inchannels, window):
        super(filter_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=inchannels, out_channels=window**2, kernel_size=3, padding=3//2, bias=False)
        self.Filterlayer = FilterLayer(window)

    def forward(self,x, f, window):
        f = self.conv(f)
        f = F.softmax(f, dim=1)
        y = self.Filterlayer(x, f, window)
        return y

class net_orig(nn.Module):
    def __init__(self, nFilter, multi=True):
        super(net_orig, self).__init__()
        self.multi = multi
        #---------------------------------------------------------------------------------------------------------
        self.pixelunshuffle_1 = PixelUnshuffle(2)
        self.conv_stride_1 = nn.Conv2d(20, nFilter * 2, 3, stride=1, padding=3 // 2)
        self.pre_block_1_1 = pre_block(nFilter * 2)
        self.conv_cat_1 = nn.Conv2d(nFilter * 2 + 3, nFilter * 2, 1)
        self.pre_block_1_2 = pre_block(nFilter * 2)

        self.pos_block_1 = pos_block(nFilter * 2)
        self.conv_f_1 = nn.Conv2d(nFilter * 2, nFilter * 4, 1)
        self.pixelshuffle_C_1 = Pixelshuffle(2)

        self.conv_irb_1 = nn.Conv2d(nFilter * 2, 12, 3, stride=1, padding=3 // 2)
        self.pixelshuffle_I_1 = Pixelshuffle(2)

        self.filter_block7 = filter_block(nFilter, 9)

        # --------------------------------------------------------------------------------------------------------
        self.zeropadding_2 = nn.ZeroPad2d(1)
        self.conv_stride_2 = nn.Conv2d(nFilter * 2, nFilter * 2, 3, stride=2)
        self.pre_block_2_1 = pre_block(nFilter * 2)
        self.conv_cat_2 = nn.Conv2d(nFilter * 2 + 3, nFilter * 2, 1)
        self.pre_block_2_2 = pre_block(nFilter * 2)

        self.pos_block_2 = pos_block(nFilter * 2)
        self.conv_f_2 = nn.Conv2d(nFilter * 2, nFilter * 4, 1)
        self.pixelshuffle_C_2 = Pixelshuffle(2)

        self.conv_irb_2 = nn.Conv2d(nFilter * 2, 12, 3, stride=1, padding=3 // 2)
        self.pixelshuffle_I_2 = Pixelshuffle(2)

        self.filter_block9 = filter_block(nFilter, 11)

        self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # --------------------------------------------------------------------------------------------------------
        self.zeropadding_3 = nn.ZeroPad2d(1)
        self.conv_stride_3 = nn.Conv2d(nFilter * 2, nFilter * 2, 3, stride=2)
        self.pre_block_3_1 = pre_block(nFilter * 2)
        self.pre_block_3_2 = pre_block(nFilter * 2)

        self.pos_block_3 = pos_block(nFilter * 2)
        self.conv_f_3 = nn.Conv2d(nFilter * 2, nFilter * 4, 1)
        self.pixelshuffle_C_3 = Pixelshuffle(2)

        self.conv_irb_3 = nn.Conv2d(nFilter * 2, 12, 3, stride=1, padding=3 // 2)
        self.pixelshuffle_I_3 = Pixelshuffle(2)

        self.filter_block11 = filter_block(nFilter, 13)

        self.upsample_3 = nn.UpsamplingBilinear2d(scale_factor=2)
        # --------------------------------------------------------------------------------------------------------

    def forward(self, x1, x2):
        output_list = []
        _x = self.pixelunshuffle_1(x1)
        t1 = F.relu(self.conv_stride_1(_x))     #8m*8m
        t1 = self.pre_block_1_1(t1)


        t2 = self.zeropadding_2(t1)
        t2 = F.relu(self.conv_stride_2(t2))       #4m*4m
        t2 = self.pre_block_2_1(t2)

        t3 = self.zeropadding_3(t2)
        t3 = F.relu(self.conv_stride_3(t3))       #2m*2m
        t3 = self.pre_block_3_1(t3)
        t3 = self.pre_block_3_2(t3)

        _t3 = self.conv_irb_3(t3)
        _t3 = self.pixelshuffle_I_3(_t3)
        _t3 = torch.add(_t3, x2)

        t3 = self.pos_block_3(t3)
        t3 = F.relu(self.conv_f_3(t3))
        t3 = self.pixelshuffle_C_3(t3)

        t3_out = self.filter_block11(_t3, t3, 13)
        t3_out_up = self.upsample_3(t3_out)
        output_list.append(t3_out)

        t2 = torch.cat((t3_out, t2), dim=1)
        t2 = F.relu(self.conv_cat_2(t2))
        t2 = self.pre_block_2_2(t2)

        _t2 = self.conv_irb_2(t2)
        _t2 = self.pixelshuffle_I_2(_t2)
        _t2 = torch.add(_t2, t3_out_up)

        t2 = self.pos_block_2(t2)
        t2 = F.relu(self.conv_f_2(t2))
        t2 = self.pixelshuffle_C_2(t2)

        t2_out = self.filter_block9(_t2, t2, 11)
        t2_out_up = self.upsample_2(t2_out)
        output_list.append(t2_out)

        t1 = torch.cat((t1,t2_out), dim=1)
        t1 = F.relu(self.conv_cat_1(t1))
        t1 = self.pre_block_1_2(t1)

        _t1 = self.conv_irb_1(t1)
        _t1 = self.pixelshuffle_I_1(_t1)
        _t1 = torch.add(_t1,t2_out_up)

        t1 = self.pos_block_1(t1)
        t1 = F.relu(self.conv_f_1(t1))
        t1 = self.pixelshuffle_C_1(t1)

        y = self.filter_block7(_t1, t1, 9)
        output_list.append(y)

        if self.multi != True:
            return y
        else:
            return output_list


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net_orig(nFilter=64,multi=False).to(device)
    summary(model, [(5, 256, 256),(3,64,64)])

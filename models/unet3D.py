import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def CBR(bn, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1,
        bias=True):
    """
    :param bn:
    :param in_channels:
    :param out_channels:
    :param mid_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param bias:
    :return: CONVOLUTION_BATCHNORM_RELU
    """
    if mid_channels == None:
        mid_channels = out_channels // 2
    if bn:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            # TODO: TALELE RELU MOGOCE NIMA SMISLA!!! GLEJ SLIKO 3DUNET, SE CONCATA RELUJANO Z NERELUJANIM!
            # nn.MaxPool3d(kernel_size=2, stride=2, padding=0) # NCDHW
            # DODAJ MAXPOOL3D
        )
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
            # nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

    return layer


def up_conv(in_channels, out_channels, kernel_size=2, stride=2, padding=0,  # 2x2x2 stride 2 upconv..
            output_padding=0, bias=True):
    layer_upscale = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding, bias=bias)
    return layer_upscale


def maxpool(ksize=2, pad=0):
    return nn.MaxPool3d(kernel_size=ksize, padding=pad)


class Simply3DUnet(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, depth, init_feature_size=32,
                 bn=True):  # max pool is 2x2x2 stride 2
        super(Simply3DUnet, self).__init__()
        # print("depth %s"%depth)
        self.bn = bn
        self.in_ch = num_in_channels
        self.out_ch = num_out_channels
        self.depth = depth
        self.init_feature_size = init_feature_size
        self.CBR_in = CBR(bn=self.bn, in_channels=self.in_ch, mid_channels=self.init_feature_size,
                          out_channels=self.init_feature_size * 2)
        self.CBR_out = CBR(bn=self.bn, in_channels=self.init_feature_size * 6, mid_channels=self.init_feature_size * 2,
                           out_channels=self.init_feature_size * 2)
        self.final_conv = nn.Conv3d(in_channels=self.init_feature_size * 2, out_channels=self.out_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.layers_down = [self.CBR_in]
        self.layers_up = [self.CBR_out]
        self.max_pools = [maxpool() for d in range(depth)]

        self.up_convolutions = list()

        for d_ in range(depth):
            d = d_ + 1  # to make it 1-index
            in_ch_down = init_feature_size * (2 ** d)  # input channel to down layer is 2*d*self.init_feature_size
            # print(in_ch_down)
            self.layers_down.append(
                CBR(bn=self.bn, in_channels=in_ch_down, mid_channels=in_ch_down * 2, out_channels=in_ch_down * 2))
            if d > 1:  # najbolj plitvega si opisal v CBR_out
                self.layers_up.append(
                    CBR(bn=self.bn, in_channels=in_ch_down * 3, mid_channels=in_ch_down * 1,
                        out_channels=in_ch_down * 1))
            self.up_convolutions.append(up_conv(in_channels=in_ch_down * 2, out_channels=in_ch_down * 2))

        self.layers_up = reversed(self.layers_up)
        self.up_convolutions = reversed(self.up_convolutions)

        self.layers_down = nn.ModuleList(self.layers_down)
        self.layers_up = nn.ModuleList(self.layers_up)
        self.max_pools = nn.ModuleList(self.max_pools)
        self.up_convolutions = nn.ModuleList(self.up_convolutions)

        # self.to(device)

    def forward(self, x):
        down_inputs = list()
        up_inputs = list()
        x_down = x
        for i, layer in enumerate(self.layers_down):
            # print("Down layer %s"%i)
            # print("down_inp: %s"%str(x_down.size()))
            x_down = layer(x_down)
            # print("after CBR: %s"%str(x_down.size()))
            down_inputs.append(x_down)
            if i != len(self.layers_down) - 1:
                x_down = self.max_pools[i](x_down)
            else:
                # print("didnt maxpool")
                pass
            # print("after maxpool: %s"%str(x_down.size()))
        down_inputs = list(reversed(down_inputs))
        # print("DOWN INPUTS SIZES:")
        for d in down_inputs:
            #print(d.size())
            pass
        x_up = x_down
        # print("UP layer number %s"%len(self.layers_up))
        for i, layer in enumerate(self.layers_up):
            # print(i)
            # if i !=  len(self.layers_up)-1:
            # print("lol")
            x_up = self.up_convolutions[i](x_up)
            # print(x_up.size())
            # print(down_inputs[i+1].size())
            x_up = torch.cat([down_inputs[i + 1], x_up], dim=1)
            # print(x_up.size())
            x_up = layer(x_up)
            # print(x_up.size())
        x_up = self.final_conv(x_up)
        # print("final shape %s"%str(x_up.size()))
        return x_up

"""class INNLOSS(nn.Module):
    def __init__(self, tightness = 0.01, mean_loss_weight = 0.5):
        super(INNLOSS, self).__init__()
        self.tightness = tightness
        self.mean_loss_weight = mean_loss_weight

    def forward(self, otpt, real, tightness = 0.01, mean_loss_weight = 0.5):  # 3 channel output - low, mid, max;; NCHWD format
        # real format NHWD (C=1)!
        low = otpt[:,0]
        mid = otpt[:,1]
        high = otpt[:,2]

        zero = torch.zeros_like(real)
        tightness = torch.tensor(self.tightness)
        mean_loss_weight = torch.tensor(mean_loss_weight).to(device)
        #a = torch.max(torch.sub(real, high).to(device), other=zero).to(device)
        loss = torch.sub(real, high)

        torch.pow(torch.max(real - high, other=zero).to(device),exponent=2).to(device) +\
               torch.pow(torch.max(low - real, zero),2) #+\
               tightness * (high - low) +\
               mean_loss_weight * self.loss_ce(mid,real)

        return loss"""
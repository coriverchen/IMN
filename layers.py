import torch
import torch.nn as nn
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

# Resnet connection
class ResidualResBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualResBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 64, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(64, input, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input, 64, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(64, input, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input, input, 3, 1, 1, bias=bias)
        #self.conv6 = nn.Conv2d(64, input, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.conv2(x1)
        x2 = self.lrelu( x + x2 )
        x3 = self.conv3(x2)
        x3 = self.lrelu(x3+x1)
        x4 = self.conv4(x3)
        x4 = self.lrelu(x4+x2)
        x5 = self.conv5(x4)

        return x5

# Conv connection
class ResidualConvBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualConvBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(32, input, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = self.conv5(x4)
        return x5

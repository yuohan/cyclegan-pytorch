import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ Residual block
    """
    def __init__(self, conv_dim):

        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, bias=False), 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(conv_dim)
        )

    def forward(self, x):
        return x + self.model(x)

class Generator(nn.Module):
    """ ResNet-based Generator
    """
    def __init__(self, in_channels, out_channels, conv_dim):

        super(Generator, self).__init__()

        n_blocks = 6

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, conv_dim, kernel_size=7, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, 2*conv_dim, kernel_size=7, bias=False),
            nn.BatchNorm2d(2*conv_dim),
            nn.ReLU(True),
            nn.Conv2d(2*conv_dim, 4*conv_dim, kernel_size=7, bias=False),
            nn.BatchNorm2d(4*conv_dim),
            nn.ReLU(True),
            nn.ModuleList([ResidualBlock(4*conv_dim) for _ in range(n_blocks)]),
            nn.ConvTranspose2d(4*conv_dim, 2*conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(2*conv_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*conv_dim, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(conv_dim, out_channels, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """ PatchGAN Discriminator
    """
    def __init__(self, in_channels, out_channels, conv_dim):

        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim, 2*conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*conv_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*conv_dim, 4*conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*conv_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*conv_dim, 4*conv_dim, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4*conv_dim),
            nn.LeakyReLU(0.2, True),
            Conv2d(conv_dim*8, 1, kernel_size=4, stride=1, paddding=1)
        )

    def forward(self, x):
        return self.model(x)
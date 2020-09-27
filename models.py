import torch
import torch.nn as nn
import torch.optim as optim

import itertools

class ResidualBlock(nn.Module):
    """ Residual block
    """
    def __init__(self, conv_dim):

        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3), 
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3),
            nn.InstanceNorm2d(conv_dim)
        )

    def forward(self, x):
        return x + self.model(x)

class Generator(nn.Module):
    """ ResNet-based Generator
    """
    def __init__(self, in_channels, out_channels, conv_dim):
        super(Generator, self).__init__()

        n_blocks = 6
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, conv_dim, kernel_size=7),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, 2*conv_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(2*conv_dim),
            nn.ReLU(True),
            nn.Conv2d(2*conv_dim, 4*conv_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(4*conv_dim),
            nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model += [ResidualBlock(4*conv_dim)]
        model += [
            nn.ConvTranspose2d(4*conv_dim, 2*conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(2*conv_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*conv_dim, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(conv_dim, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """ PatchGAN Discriminator
    """
    def __init__(self, in_channels, conv_dim):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim, 2*conv_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(2*conv_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*conv_dim, 4*conv_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(4*conv_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4*conv_dim, 8*conv_dim, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(8*conv_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class GeneratorLoss(nn.Module):
    """ Generator Loss log(D(G(x)))
    """
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_fake):
        return self.loss(pred_fake, torch.ones_like(pred_fake))

class DiscriminatorLoss(nn.Module):
    """ Discriminator Loss log(D(y) + log(1-D(G(x)))
    """
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake):
        return (self.loss(pred_real, torch.ones_like(pred_real)) + self.loss(pred_fake, torch.zeros_like(pred_fake))) * 0.5

class CycleGAN(nn.Module):
    """ Cycle GAN 
    """
    def __init__(self, in_channels, out_channels, conv_dim=64, lr=2e-4, cycle_weight=10):
        super(CycleGAN, self).__init__()

        self.G_X = Generator(in_channels, out_channels, conv_dim)
        self.G_Y = Generator(in_channels, out_channels, conv_dim)

        self.D_X = Discriminator(in_channels, conv_dim)
        self.D_Y = Discriminator(in_channels, conv_dim)

        self.criterion_G = GeneratorLoss()
        self.criterion_D = DiscriminatorLoss()
        self.criterion_C = nn.L1Loss()

        self.optim_G = optim.Adam(itertools.chain(self.G_X.parameters(), self.G_Y.parameters()), lr, beta=[0.5, 0.999])
        self.optim_Y = optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr, beta=[0.5, 0.999])
        
        self.cycle_weight = cycle_weight

    def train(self, real_X, real_Y):

        fake_X = self.G_Y(real_Y) # G_Y(Y)
        fake_Y = self.G_X(real_X) # G_X(X)

        cycle_X = self.G_Y(fake_Y) # G_Y(G_X(X))
        cycle_Y = self.G_X(fake_X) # G_X(G_Y(Y))

        # Loss = L_gen(G, D_Y, X, Y) + L_gen(F, D_X, Y, X) + lambda * L_cyc(G, F)
        # L_gen(G, D, X, Y) = E[logD(y)] + E[1-logD(G(x))]

        # Train the generator
        # Disable gradient of Discriminators
        for model in [self.D_X, self.D_Y]:
            for param in model.parameters():
                param.requires_grad = False
        self.optim_G.zero_grad()

        # maximize log(D(G(x))) instead of log(1-D(G(x)))
        G_X_loss = self.criterion_G(self.D_X(fake_Y))
        G_Y_loss = self.criterion_G(self.D_Y(fake_X))
        # Cycle loss ||G_B(G_A(A)) - A|| + ||G_A(G_B(B)) - B||
        cycle_loss = self.criterion_C(cycle_X, real_X) + self.criterion_C(cycle_Y, real_Y)
        G_loss = G_X_loss + G_Y_loss + self.cycle_weight*cycle_loss
        G_loss.backward()
        self.optim_G.step()

        # Train the discriminator
        for model in [self.D_X, self.D_Y]:
            for param in model.parameters():
                param.requires_grad = True
        self.optim_D.zero_grad()
        
        # pool from fake_X, fake_Y
        D_X_loss = self.criterion_D(self.D_X(real_Y), self.D_X(fake_Y.detach()))
        D_Y_loss = self.criterion_D(self.D_Y(real_X), self.D_Y(fake_X.detach()))

        D_X_loss.backward()
        D_Y_loss.backward()
        self.optim_D.step()

        return G_loss, (D_X_loss + D_Y_loss)
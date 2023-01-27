import torch.nn as nn


class DCGAN_With_Tanh_Leaky(nn.Module):

    def __init__(self, latent_dim=100, image_size=64, color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*2, image_size*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*4, image_size*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


class DCGAN_With_Sigmoid_Leaky(nn.Module):

    def __init__(self, latent_dim=100, image_size=64, color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Sigmoid()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*2, image_size*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*4, image_size*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


class DCGAN_With_Tanh_ReLU(nn.Module):

    def __init__(self, latent_dim=100, image_size=64, color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*2, image_size*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*4, image_size*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


class DCGAN_With_Sigmoid_ReLU(nn.Module):

    def __init__(self, latent_dim=100, image_size=64, color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Sigmoid()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*2, image_size*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*4, image_size*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


class DCGAN_With_Tanh_Both(nn.Module):

    def __init__(self, latent_dim=100, image_size=64, color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*2, image_size*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*4, image_size*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size*8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


class DCGAN_With_Tanh_ReLU_Leaky(nn.Module):

    def __init__(self, latent_dim=100, image_size=64, color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*2, image_size*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*4, image_size*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(image_size*8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(image_size*8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits

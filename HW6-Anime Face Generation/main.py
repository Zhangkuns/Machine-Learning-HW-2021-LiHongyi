# **Homework 6 - Generative Adversarial Network**
"""
This is the example code of homework 6 of the machine learning course by Prof. Hung-yi Lee.
In this homework, you are required to build a generative adversarial  network for anime face generation.
"""
import random
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from qqdm.notebook import qqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging

# You may replace the workspace directory if you want.
workspace_dir = '.'


# Random seed
# Set the random seed to a certain value for reproducibility.

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2021)

# Dataset
'''
1. Resize the images to (64, 64)
1. Linearly map the values from [0, 1] to  [-1, 1].

Please refer to [PyTorch official website](https://pytorch.org/vision/stable/transforms.html) for details about 
different transforms. '''


# prepare for CrypkoDataset

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


# Show some images
'''Note that the values are in the range of [-1, 1], we should shift them to the valid range, [0, 1], to display 
correctly. '''

dataset = get_dataset(os.path.join(workspace_dir, 'faces'))

images = [dataset[i] for i in range(4)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

'''Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).'''
images = [(dataset[i] + 1) / 2 for i in range(16)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

# Model
'''
In this section, we will create models and trainer.
Here, we use DCGAN as the model structure. Feel free to modify your own model structure.
Note that the `N` of the input/output shape stands for the batch size.
'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Generator

class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            """torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
            output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

            Applies a 2D transposed convolution operator over an input image composed of several input planes.

            This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a
            fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation
            as it does not compute a true inverse of convolution).

            in_channels (int) – Number of channels in the input image
            out_channels (int) – Number of channels produced by the convolution
            kernel_size (int or tuple) – Size of the convolving kernel
            stride – Stride of the convolution. Default: 1 controls the stride for the cross-correlation.
            padding – controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1)
                      padding number of points.
            output_padding (int or tuple, optional) – Additional size added to one side of each dimension in the output
                                                      shape. Default: 0
            groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
                                     controls the connections between inputs and outputs.
            bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
            dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1 controls the spacing
                                                between the kernel points; also known as the à trous algorithm.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),  # (batch, feature_dim * 16, 8, 8)
            dconv_bn_relu(dim * 4, dim * 2),  # (batch, feature_dim * 16, 16, 16)
            dconv_bn_relu(dim * 2, dim),  # (batch, feature_dim * 16, 32, 32)
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),  # double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


# Discriminator

class Discriminator(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch, )
    """

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        # input: (batch, 3, 64, 64)
        """
        NOTE FOR SETTING DISCRIMINATOR:
        Remove last sigmoid layer for WGAN
        """

        def conv_bn_lrelu(in_dim, out_dim):
            """
            NOTE FOR SETTING DISCRIMINATOR:
            You can't use nn.Batchnorm for WGAN-GP
            Use nn.InstanceNorm2d instead
            """
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        """ Medium: Remove the last sigmoid layer for WGAN. """
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, kernel_size=4,stride=1,padding=0),
            # nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


# Training
# Creat trainer
# Initialization
# - hyperparameters
# - model
# - optimizer
# - dataloader

# Training hyperparameters
batch_size = 64
z_dim = 100
z_sample = Variable(torch.randn(100, z_dim)).cuda()
lr = 1e-4

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = 1  # 50
n_critic = 1  # 5
clip_value = 0.01

log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
G = Generator(in_dim=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# Loss
'''
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
'''
criterion = nn.BCELoss()

""" 
Medium: Use RMSprop for WGAN.
NOTE FOR SETTING OPTIMIZER:
     GAN: use Adam optimizer
     WGAN: use RMSprop optimizer
     WGAN-GP: use Adam optimizer 
"""
# Optimizer
# opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Training loop
'''We store some pictures regularly to monitor the current performance of the Generator, 
   and regularly record checkpoints.'''

steps = 0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar = tqdm(dataloader)
    progress_bar.set_description(f"Epoch {e + 1}")
    for i, data in enumerate(progress_bar):
        imgs = data.cuda()
        bs = imgs.size(0)

        # ============================================
        #  Train D
        # ============================================
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)

        """ Medium: Use WGAN Loss. """
        # Label
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()

        # Discriminator forwarding
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())

        # Compute the loss for the discriminator.
        # WGAN Loss
        # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
        """
         NOTE FOR SETTING DISCRIMINATOR LOSS:
         GAN: 
            loss_D = (r_loss + f_loss)/2
         WGAN: 
            loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
         WGAN-GP: 
            gradient_penalty = self.gp(r_imgs, f_imgs)
            loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
         """
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = -torch.mean(r_logit) + torch.mean(f_logit)

        # Discriminator backwarding
        D.zero_grad()
        loss_D.backward()

        # Update the discriminator.
        opt_D.step()

        """ Medium: Clip weights of discriminator. """
        '''torch.clamp(input, min=None, max=None, *, out=None) 
           Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, '''
        for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # ============================================
        #  Train G
        # ============================================
        if steps % n_critic == 0:
            # Generate some fake images.
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            # Model forwarding
            f_logit = D(f_imgs)

            """ Medium: Use WGAN Loss"""
            '''
            NOTE FOR SETTING LOSS FOR GENERATOR:
            GAN: loss_G = self.loss(f_logit, r_label)
            WGAN: loss_G = -torch.mean(self.D(f_imgs))
            WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
            '''
            # Compute the loss for the generator.
            # WGAN Loss
            loss_G = -torch.mean(D(f_imgs))

            # Generator backwarding
            G.zero_grad()
            loss_G.backward()

            # Update the generator.
            opt_G.step()

            if steps % 10 == 0:
                progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                # Set/modify postfix (additional stats) with automatic formatting based on datatype.

        steps += 1

    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')

    # Show generated images during training.
    grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    G.train()

    if (e + 1) % 5 == 0 or e == 0:
        # Save the checkpoints.
        torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

logging.info('Finish training')

# Inference
# Use the trained model to generate anime faces!

# Load model
G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
G.eval()
G.cuda()

# Generate and show some images.

# Generate 1000 images and make a grid to save them.
n_output = 1000
z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0

# Save the generated images.
os.makedirs('output', exist_ok=True)
for i in range(n_output):
    torchvision.utils.save_image(imgs[i], f'output/{i + 1}.jpg')

log_dir = os.path.join(workspace_dir, 'logs')
filename = os.path.join(log_dir, 'result.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=10)

# Show 32 of the images.
grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()





import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
 
        #ulaz 6 256 256
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1, bias=False) 
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True) 

        #ulaz 64 128 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        #ulaz 128 64 64 
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        #ulaz 256 32 32
        self.zero_pad1 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False) # ulaz 256 34 34
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        #ulaz 512 31 31
        self.zero_pad2 = nn.ZeroPad2d(1) 
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1)  #ulaz 512 33 33
        # izlaz 1 30 30

    def forward(self, input_img, target_img):
        # 1 3 h w i 1 3 h w
        
        if input_img.dim() == 3:
            input_img = input_img.unsqueeze(0)
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(0)

        x = torch.cat([input_img, target_img], dim=1)  # (1, 6, H, W)

        x = self.conv1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leaky_relu2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leaky_relu3(x)

        x = self.zero_pad1(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.leaky_relu4(x)

        x = self.zero_pad2(x)
        x = self.conv5(x)

        x = x.squeeze(0)  # ukloni batch > (1, 30, 30)

        return x

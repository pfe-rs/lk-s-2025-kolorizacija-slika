import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from BaseColor import *

class Generator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(Generator, self).__init__()

        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(True),
            norm_layer(64)
        ]

        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(True),
            norm_layer(128)
        ]

        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(True),
            norm_layer(256)
        ]

        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            norm_layer(512)
        ]

        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            norm_layer(512)
        ]

        model6 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)

        self.softmax = nn.Softmax(dim=1)
        #self.tanh = nn.Tanh()
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, dilation=1, stride=1, padding=0, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
    def forward(self, input_l, return_logits=False):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        logits = self.model6(conv5_3)
        # ab_pred = self.model_out(self.softmax(logits))           # [B, 2, H/4, W/4]
        # ab_pred = self.unnormalize_ab(self.upsample4(ab_pred))   # [B,2,H,W]
        ab_pred = self.model_out(self.softmax(logits))
  # linearna kombinacija binova
        #ab_pred = self.tanh(ab_pred) * 110           # skaliranje u realan Lab prostor
        ab_pred = self.upsample4(ab_pred)          # do pune rezolucije
        ab_pred = self.unnormalize_ab(ab_pred)     # ako koristiš BaseColor normu


        if return_logits:
            return ab_pred, logits
        else:
            return ab_pred

                
    def colorizer(pretrained=True):
        model = Generator()
        if(pretrained):
            import torch.utils.model_zoo as model_zoo
            model.load_state_dict(
                model_zoo.load_url(
                    'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
                    map_location='cpu',
                    check_hash=True)
                )
            return model
    

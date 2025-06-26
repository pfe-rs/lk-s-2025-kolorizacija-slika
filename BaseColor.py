import torch
from torch import nn
import numpy as np
from PIL import Image
from skimage import color
import torch.nn.functional as F

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm
	
def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=4):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=4):
    basecolor = BaseColor()

    img_rgb = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab = color.rgb2lab(img_rgb).astype(np.float32)

    img_l = img_lab[:,:,0]
    img_ab = img_lab[:,:,1:3]

    tens_l = torch.tensor(img_l)[None, None, :, :]
    tens_l = basecolor.normalize_l(tens_l)

    tens_ab = torch.tensor(img_ab).permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
    tens_ab = basecolor.normalize_ab(tens_ab)

    return tens_l, tens_ab, img_lab


def postprocess_img(tens_l, tens_gen_ab, mode='bilinear'):

    #tens l: tensor oblika (1, 1, H_orig, W_orig)
   # out_ab: tensor oblika (1, 2, H, W)
   # Vraca: tensor oblika (1, 3, H_orig, W_orig) -> L + ab (resized)
    basecolor = BaseColor()
    tens_l = basecolor.unnormalize_l(tens_l)  # L kanal
    tens_gen_ab = basecolor.unnormalize_ab(tens_gen_ab)
    HW_orig = tens_l.shape[2:]
    HW_ab = tens_gen_ab.shape[2:]

    # skaliranje ab na l hw
    if HW_ab != HW_orig:
        out_ab_resized = F.interpolate(tens_gen_ab, size=HW_orig, mode=mode, align_corners=False)
    else:
        out_ab_resized = tens_gen_ab

    lab_tensor = torch.cat([tens_l, out_ab_resized], dim=1) #  # Spoji L i ab kanale u 1 3 h w
    return lab_tensor

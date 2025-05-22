
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

def load_img(img_path): # xử lý đầu vào ra ra của ảnh
	out_np = np.asarray(Image.open(img_path)) # chuyển ảnh về dạng numpy
	if(out_np.ndim==2): # Nếu ảnh chỉ có 2 chiều (ảnh grayscale)
		out_np = np.tile(out_np[:,:,None],3) # Nhân thành 3 kênh để giả lập RGB
	return out_np

def resize_img(img, HW=(256,256), resample=3): # resize ảnh về kích thước 256x256
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig) # Chuyển gốc ảnh sang LAB
	img_lab_rs = color.rgb2lab(img_rgb_rs) # Chuyển ảnh resize sang LAB

	img_l_orig = img_lab_orig[:,:,0] # Lấy kênh L (Lightness – độ sáng)
	img_l_rs = img_lab_rs[:,:,0]  # Lấy kênh L sau resize

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]  # 1 x 1 x H x W (tensor gốc)
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]   # 1 x 1 x 256 x 256 (tensor resize)

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]	# Kích thước ảnh gốc
	HW = out_ab.shape[2:]		 # Kích thước đầu ra (thường là 64x64)

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):  # Nếu khác kích thước
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear') # Resize lại ab
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1) #  Ghép kênh L và ab lại (1 x 3 x H x W)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

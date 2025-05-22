import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='src/imgs/scale.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with eccv16.png suffix')
opt = parser.parse_args()

# Load hình tô màu
colorizer_eccv16 = eccv16(pretrained=True).eval() # Load mô hình ECCV16, chuyển sang chế độ eval
if(opt.use_gpu):
    colorizer_eccv16.cuda() # Nếu dùng GPU thì chuyển mô hình lên GPU

img = load_img(opt.img_path)    # Đọc ảnh RGB
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256)) # Lấy kênh L (Lightness), bản gốc và bản resize 256x256
if(opt.use_gpu):
    tens_l_rs = tens_l_rs.cuda() # Nếu dùng GPU, đưa tensor vào GPU


img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
                # Tạo ảnh đen trắng
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
                # Gọi mô hình để tô màu

plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16) # Lưu ảnh tô màu ra file


# Out put ra màn hình
plt.figure(figsize=(8,4)) # Tạo khung hình có khung ngang 8inch và cao 4 inch
plt.subplot(1,2,1) # Chia màn hình thành 1 hàng 2 cột
plt.imshow(img_bw) # Dán ảnh trắng đen lên
plt.title('Input') # Tiêu đề
plt.axis('off') # Xóa trục xy

plt.subplot(1,2,2) # Cột bên phải
plt.imshow(out_img_eccv16) # Vẽ ảnh đầu ra
plt.title('Output') # Tiêu đề out put
plt.axis('off') # Xóa trục x y
plt.show() # Show ra màn hình

import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50. # Giá trị trung tâm cho kênh L
		self.l_norm = 100. # Giá trị chuẩn hóa cho Kênh L
		self.ab_norm = 110. # Giá trị chuẩn hóa cho 2 kênh ab

	def normalize_l(self, in_l):  # Hàm chuẩn hóa kênh L giá trị sau khi tính xong từ [-0.5,0.5]
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l): # Hàm này chuyển cái chuẩn hóa vừa rồi thành về lại giá trị kênh L như cũ
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab): # Hầm chuẩn hóa kênh màu a và b giá trị dao động từ [-1,1]
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab): # Hàm hôi chuẩn kênh ab về như cũ
		return in_ab*self.ab_norm


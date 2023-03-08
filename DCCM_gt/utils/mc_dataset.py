from torch.utils.data import DataLoader, Dataset
import numpy as np

# from skimage import io, color 暂时用PIL代替
from PIL import Image

import torchvision.transforms as transforms
import torch
import pdb

class McDataset(Dataset):
	def __init__(self, root_dir, meta_file, transform=None):
		# 数据集目录
		self.root_dir = root_dir
		# 采用的变换
		self.transform = transform
		# 打开元文件
		with open(meta_file) as f:
			lines = f.readlines()
		print("building dataset from %s" % meta_file)
		# 获取行数（对于图片数量）
		self.num = len(lines)
		self.metas = []
		self.imgs = []
		for line in lines:
			# 获取路径和类别
			path, cls = line.rstrip().split()
			# 形成一个元组（路径，类别）
			self.metas.append((path, int(cls)))
			# 保存对应的图片（目录，类别）
			self.imgs.append((self.root_dir + '/' + path, int(cls)))
 
	def __len__(self):
		return self.num

	def __getitem__(self, idx):
		filename = self.root_dir + '/' + self.metas[idx][0]
		cls = self.metas[idx][1]
		# img = io.imread(filename)
		# img = color.gray2rgb(img)
		img = Image.open(filename).convert('RGB')
		if self.transform:
			img = self.transform(img)
		return img, cls


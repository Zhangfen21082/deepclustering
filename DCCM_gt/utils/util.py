import os
import pickle
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
from .myLinearAssignment import linear_assignment
import pdb

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, length=0):
		self.length = length
		self.reset()

	def reset(self):
		if self.length > 0:
			self.history = []
		else:
			self.count = 0
			self.sum = 0.0
		self.val = 0.0
		self.avg = 0.0

	def update(self, val):
		if self.length > 0:
			self.history.append(val)
			if len(self.history) > self.length:
				del self.history[0]
			self.val = self.history[-1]
			self.avg = np.mean(self.history)
		else:
			self.val = val
			self.sum += val
			self.count += 1
			self.avg = self.sum / self.count

def learning_rate_decay(optimizer, t, lr_0):
	for param_group in optimizer.param_groups:
		lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
		param_group['lr'] = lr


class Logger():
	""" Class to update every epoch to keep trace of the results
	Methods:
		- log() log and save
	"""

	def __init__(self, path):
		self.path = path
		self.data = []

	def log(self, train_point):
		self.data.append(train_point)
		with open(os.path.join(self.path), 'wb') as fp:
			pickle.dump(self.data, fp, -1)

"""
create_logger用于创建一个日志记录器对象，四个参数含义如下
	-name：日志记录器对象名称
	-log_file：日志文件的路径
	-rank：日志记录器对象的登记，默认为0级
	-level：日志记录器对象的记录级别，默认为logging.INFO
函数返回创建好的日志记录器对象 l。可以通过调用这个返回的日志记录器对象的 debug()、info()、warning()、error() 等方法记录日志
日志会同时输出到指定的日志文件和控制台
"""
def create_logger(name, log_file, rank=0, level=logging.INFO):
	# 使用logging.getLogger()方法创建一个名字为name的日志记录器对象 l
	l = logging.getLogger(name)
	# 使用logging.Formatter()方法创建一个格式化器，格式化器格式包含时间戳、文件名、行号、日志级别和排名
	# 其中排名会根据传入的rank参数进行格式化
	formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s][rank:{}] %(message)s'.format(rank))
	# 使用logging.FileHandler和logging.StreamHandler方法创建两个处理器对象fh和sh分别用于将日志记录到文件控制台
	# 通过 fh.setFormatter() 和 sh.setFormatter() 方法将格式化器 formatter 应用到处理器对象 fh 和 sh 中
	fh = logging.FileHandler(log_file)
	fh.setFormatter(formatter)
	sh = logging.StreamHandler()
	sh.setFormatter(formatter)
	# 设置日志记录器对象的记录级别为level，并向日志记录器对象l中添加处理器对象fh和sh
	l.setLevel(level)
	l.addHandler(fh)
	l.addHandler(sh)
	return l

def clustering_acc(y_true, y_pred):
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	ind = linear_assignment(w.max() - w)

	return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

"""
实现加权二元交叉熵损失（WBCE）
inputs: 是模型的输出，表示每个样本属于正类的概率。
targets: 是真实标签，0或1。
weights: 是每个样本的权重。
"""
class WeightedBCE(nn.Module):

	def __init__(self, eps=1e-12, use_gpu=True):
		super(WeightedBCE, self).__init__()
		self.eps = eps
		self.use_gpu = use_gpu

	def forward(self, inputs, targets, weights):
		log_probs_pos = torch.log(inputs + self.eps)
		log_probs_neg = torch.log(1 - inputs + self.eps)
		loss1 = - targets * log_probs_pos
		loss2 = -(1 - targets) * log_probs_neg
		loss3 = loss1 + loss2
		loss4 = loss3.mean(1)
		loss5 = weights * loss4
		loss = loss5.mean()		
	
		return loss

# 加载断点
def load_checkpoint(model, dim_loss, classifier, optimizer, ckpt_path):

	checkpoint = torch.load(ckpt_path)
	
	model.load_state_dict(checkpoint['model'])
	dim_loss.load_state_dict(checkpoint['dim_loss'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	best_nmi = checkpoint['best_nmi']
	start_epoch = checkpoint['epoch']

	return start_epoch, best_nmi

# 保存断点
def save_checkpoint(state, is_best_nmi, filename):
	torch.save(state, filename+'.pth.tar')
	if is_best_nmi:
		shutil.copyfile(filename+'.pth.tar', filename+'_best_nmi.pth.tar')
	
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


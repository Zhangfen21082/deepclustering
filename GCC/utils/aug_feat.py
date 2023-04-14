"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import json
import torch
import pickle
import numpy as np

"""
这个类名为AugFeat，它的作用是实现一个特征缓存器，用于以内存高效的方式存储特征
具体来说，它可以将特征向量和对应的索引存储在内存字典中，、
当内存字典中的特征向量数量达到一定阈值时，会将最早的特征向量弹出，
以便存储新的特征向量。此外，该类还提供了一个保存内存字典的方法。

- output_path：指定内存字典保存的路径。
- size：指定内存字典中每个键对应的特征向量数量的最大值。
- alpha：指定内存字典中每个键对应的特征向量的加权平均系数。
"""
class AugFeat:
    def __init__(self, output_path, size=20, alpha=0.3):
        self.memory_dict = {}
        self.output_path = output_path
        self.size = size
        self.alpha = alpha

    @torch.no_grad()
    def push(self, feats, indexes):
        for i in range(indexes.shape[0]):
            key = indexes[i].item()
            #data = feats[i][0].cpu().detach().numpy()
            data = feats[i].unsqueeze_(0)
            if key not in self.memory_dict.keys():
                self.memory_dict[key] = [data]
                #self.memory_dict[key] = data
            elif len(self.memory_dict[key]) == self.size:
                self.memory_dict[key].pop(0)
                self.memory_dict[key].append(data)
            else:
                self.memory_dict[key].append(data)
                #self.memory_dict[key] = self.alpha * self.memory_dict[key] + (1.0 - self.alpha) * data

    @torch.no_grad()
    def pop(self, key):
        return self.memory_dict[key]

    def save(self):
        np.save(self.output_path, self.memory_dict)

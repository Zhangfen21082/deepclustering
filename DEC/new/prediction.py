from Tmnist_dec import CachedMNIST
from DEC.dec import DEC
from DEC.dec_train import train, predict
from DEC.dec_utils import cluster_accuracy
import torch


model = torch.load('./model_save/model.pth', map_location='cpu')

ds_val = CachedMNIST(
    train=False, cuda=False, testing_mode=False
)

predicted, actual = predict(
    ds_val, model, 1024, silent=True, return_actual=True, cuda=False
)

actual = actual.cpu().numpy()
predicted = predicted.cpu().numpy()
reassignment, accuracy = cluster_accuracy(actual, predicted)
print("Final DEC accuracy: %s" % accuracy)
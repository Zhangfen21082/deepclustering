import click
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid


from DEC.dec import DEC
from DEC.dec_train import train, predict
from DEC.dec_utils import cluster_accuracy

from SADE.sdae import StackedDenoisingAutoEncoder
import SADE.sade_train as ae


# 缓存数据用于加载
class CachedMNIST(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        # 每张图像需要进行函数_transformation所描述的转换
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST(r'E:\Postgraduate\DateSet\uncompressed', download=True, train=train, transform=img_transform)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()


    """
    其实可以用torch.ToTensor()代替

    作用：此函数将图像img转化为一个浮点数张量，并且再乘以0.02，主要目的是将图像的像素值缩小到一个较小的范围内
          便于训练

    img.tobytes()：将图像转化为字节序列
    torch.ByteStorge.from_buffer()：将字节序列转化为字节存储
    torch.ByteTensor：将字节存储转化为字节张量（ByteTensor）
    float：转化为浮点数张量

    """

    @staticmethod
    def _transformation(img):
        return (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
            * 0.02
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = torch.tensor(
                    self._cache[index][1], dtype=torch.long
                ).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)


# 命令行参数
@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=300,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=500,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):

    # 训练数据
    ds_train = CachedMNIST(
        train=True, cuda=cuda, testing_mode=testing_mode
    )
    # 验证数据
    ds_val = CachedMNIST(
        train=False, cuda=cuda, testing_mode=testing_mode
    )
    # 堆叠降噪自编码器
    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    autoencoder.load_state_dict(torch.load('./model_save/autoencoder.pth'))

    # DEC 阶段
    print("DEC stage.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()

    if os.path.isfile('./model_save/model.pth'):
        model.load_state_dict(torch.load('./model_save/model.pth'))
    else:
        dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        train(
            dataset=ds_train,
            model=model,
            epochs=100,
            batch_size=256,
            optimizer=dec_optimizer,
            stopping_delta=0.000001,
            cuda=cuda,
        )

        torch.save(model.state_dict(), './model_save/model.pth')


    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )

    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)




if __name__ == "__main__":
    main()

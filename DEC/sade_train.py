from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dae import DenoisingAutoencoder
from sdae import StackedDenoisingAutoEncoder


def train(
    dataset: torch.utils.data.Dataset,
    autoencoder: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    validation: Optional[torch.utils.data.Dataset] = None,
    corruption: Optional[float] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
    """
    使用给定的数据集训练自编码器，如果数据集由（特征, 预测）组成，则会去除预测

    :param dataset: 训练数据集，形状为[batch_size, feature]
    :param autoencoder: 需要训练的自编码器
    :param epochs: 训练epochs轮
    :param batch_size: batch_size大小
    :param optimizer: 所使用的优化器
    :param scheduler: 所用的Scheduler，默认为None
    :param corruption: 输入的噪声比例，默认为None
    :param validation: 验证集，默认为None
    :param cuda: 是否使用GPU, 默认True
    :param sampler: Dataloader的采样器, 默认为None
    :param silent: 如果为True则不打印统计信息, 默认为False
    :param update_freq: 计步器频率, 设置为None表示禁用，默认为1
    :param update_callback: 可选的损失函数和验证损失函数
    :param num_workers: 数据加载的线程数量
    :param epoch_callback: epoch和model的可选函数
    :return: None
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers if num_workers is not None else 0,
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False,
        )
    else:
        validation_loader = None
    loss_function = nn.MSELoss()
    autoencoder.train()
    validation_loss_value = -1
    loss_value = 0
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit="batch",
            postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1,},
            disable=silent,
        )
        for index, batch in enumerate(data_iterator):
            if (
                isinstance(batch, tuple)
                or isinstance(batch, list)
                and len(batch) in [1, 2]
            ):
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            # 送入自编码器
            if corruption is not None:
                output = autoencoder(F.dropout(batch, corruption))
            else:
                output = autoencoder(batch)
            loss = loss_function(output, batch)
            # accuracy = pretrain_accuracy(output, batch)
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
            )
        if update_freq is not None and epoch % update_freq == 0:
            if validation_loader is not None:
                validation_output = predict(
                    validation,
                    autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=True,
                    encode=False,
                )
                validation_inputs = []
                for val_batch in validation_loader:
                    if (
                        isinstance(val_batch, tuple) or isinstance(val_batch, list)
                    ) and len(val_batch) in [1, 2]:
                        validation_inputs.append(val_batch[0])
                    else:
                        validation_inputs.append(val_batch)
                validation_actual = torch.cat(validation_inputs)
                if cuda:
                    validation_actual = validation_actual.cuda(non_blocking=True)
                    validation_output = validation_output.cuda(non_blocking=True)
                validation_loss = loss_function(validation_output, validation_actual)
                # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                validation_loss_value = float(validation_loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.6f" % loss_value,
                    vls="%.6f" % validation_loss_value,
                )
                autoencoder.train()
            else:
                validation_loss_value = -1
                # validation_accuracy = -1
                data_iterator.set_postfix(
                    epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                )
            if update_callback is not None:
                update_callback(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_value,
                    validation_loss_value,
                )
        if epoch_callback is not None:
            autoencoder.eval()
            epoch_callback(epoch, autoencoder)
            autoencoder.train()


def pretrain(
    dataset,
    autoencoder: StackedDenoisingAutoEncoder,
    epochs: int,
    batch_size: int,
    optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
    scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
    validation: Optional[torch.utils.data.Dataset] = None,
    corruption: Optional[float] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
    """


    给定一个自编码器，使用数据集中提供的数据对其进行训练,为简便起见，仅在训练数据集上显示准确率
    如果数据集由（特征, 预测）组成，则会去除预测

    :param dataset: 训练数据集，形状为[batch_size, feature]
    :param autoencoder: 需要训练的自编码器
    :param epochs: 训练epochs轮
    :param batch_size: batch_size大小
    :param optimizer: 所使用的优化器
    :param scheduler: 所用的Scheduler，默认为None
    :param corruption: 输入的噪声比例，默认为None
    :param validation: 验证集，默认为None
    :param cuda: 是否使用GPU, 默认True
    :param sampler: Dataloader的采样器, 默认为None
    :param silent: 如果为True则不打印统计信息, 默认为False
    :param update_freq: 计步器频率, 设置为None表示禁用，默认为1
    :param update_callback: 可选的损失函数和验证损失函数
    :param num_workers: 数据加载的线程数量
    :param epoch_callback: epoch和model的可选函数
    :return: None
    """
    current_dataset = dataset
    current_validation = validation
    number_of_subautoencoders = len(autoencoder.dimensions) - 1
    for index in range(number_of_subautoencoders):
        # 拿到编码器和解码器
        encoder, decoder = autoencoder.get_stack(index)
        embedding_dimension = autoencoder.dimensions[index]
        hidden_dimension = autoencoder.dimensions[index + 1]
        # manual override to prevent corruption for the last subautoencoder
        if index == (number_of_subautoencoders - 1):
            corruption = None
        # 构造降噪编码器
        sub_autoencoder = DenoisingAutoencoder(
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            activation=torch.nn.ReLU()
            if index != (number_of_subautoencoders - 1)
            else None,
            corruption=nn.Dropout(corruption) if corruption is not None else None,
        )
        if cuda:
            sub_autoencoder = sub_autoencoder.cuda()
        ae_optimizer = optimizer(sub_autoencoder)
        ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler
        train(
            current_dataset,
            sub_autoencoder,
            epochs,
            batch_size,
            ae_optimizer,
            validation=current_validation,
            corruption=None,  # already have dropout in the DAE
            scheduler=ae_scheduler,
            cuda=cuda,
            sampler=sampler,
            silent=silent,
            update_freq=update_freq,
            update_callback=update_callback,
            num_workers=num_workers,
            epoch_callback=epoch_callback,
        )
        # copy the weights
        sub_autoencoder.copy_weights(encoder, decoder)
        # 通过子自动编码器的编码器部分传递数据集
        if index != (number_of_subautoencoders - 1):
            current_dataset = TensorDataset(
                predict(
                    current_dataset,
                    sub_autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=silent,
                )
            )
            if current_validation is not None:
                current_validation = TensorDataset(
                    predict(
                        current_validation,
                        sub_autoencoder,
                        batch_size,
                        cuda=cuda,
                        silent=silent,
                    )
                )
        else:
            current_dataset = None  # minor optimisation on the last subautoencoder
            current_validation = None


def predict(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    cuda: bool = True,
    silent: bool = False,
    encode: bool = True,
) -> torch.Tensor:
    """
    给定一个数据集，以eval模式运行模型，并串联输出。

    :param dataset: 评估数据集
    :param model: 用于预测的模型
    :param batch_size: batch size大小
    :param cuda: GPU，默认为True
    :param silent: 是否显示，默认为False
    :param encode: 是只使用编码器还是整个自编码器
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        batch = batch.squeeze(1).view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        features.append(
            output.detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features)

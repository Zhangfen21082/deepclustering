"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


"""
根据传入的环境配置文件路径和实验配置文件路径，返回一个配置字典。具体来说，
它首先从环境配置文件中读取root_dir，然后从实验配置文件中读取配置信息，并将其复制到一个新的字典cfg中。
接着，它设置了一些路径，如预训练模型的路径、特征存储路径、日志文件路径等。
如果需要进行聚类或自标签步骤，还会设置一些额外的路径。最后，它返回了这个配置字典cfg
"""
def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    end2end_dir = os.path.join(base_dir, 'end2end')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(end2end_dir)
    cfg['end2end_dir'] = end2end_dir
    cfg['end2end_checkpoint'] = os.path.join(end2end_dir, 'checkpoint.pth.tar')
    cfg['end2end_model'] = os.path.join(end2end_dir, 'model.pth.tar')
    cfg['features'] = os.path.join(end2end_dir, 'features')
    cfg['topk_neighbors_train_path'] = os.path.join(end2end_dir, 'topk-train-neighbors.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(end2end_dir, 'topk-val-neighbors.npy')
    cfg['log_output_file'] = os.path.join(end2end_dir, 'log.txt')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        selflabel_dir = os.path.join(base_dir, 'selflabel')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        mkdir_if_missing(selflabel_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')

    return cfg

import imp
from torch.utils import *
from torch import nn
import torch,math,os
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from dataset import DatasetModule
from model import ModelModule

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    ModelModule.add_model_specific_args(parser)
    DatasetModule.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    
    parser.add_argument("--config_file", type=str, default = "DPGN/config/5way_1shot_resnet12_cifar-fs.py")
    args = parser.parse_args()

    config_file = args.config_file
    config = imp.load_source("", config_file).config
    train_opt = config['train_config']
    eval_opt = config['eval_config']
    train_opt['num_queries'] = 1
    eval_opt['num_queries'] = 1

    trainer = pl.Trainer.from_argparse_args(args)
    pl_model = ModelModule(train_opt=train_opt,config=config,eval_opt=eval_opt,**args.__dict__)
    data = DatasetModule(train_opt=train_opt,**args.__dict__)
    trainer.fit(pl_model,data)

    
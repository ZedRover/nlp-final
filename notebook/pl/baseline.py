import sys
sys.path.append('/lustre/home/acct-aemwx/aemwx-user1/wangyu/my/nlp-final')
import numpy as np
import pandas as pd
import config
import torch
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pl_bolts.datamodules import SklearnDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torchmetrics as tm
import warnings
warnings.filterwarnings('ignore')
from models.model import BaseLineModel
gpu = False
random.seed(999)

wandb.login()
wandb_logger = WandbLogger(project='baseline')

class MyDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = config.DATA_DIR, batch_size: int = 64, num_workers: int = 4,fraction_rate: float = 0.8,val_fraction_rate: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10
        self.fraction_rate = fraction_rate
        self.val_fraction_rate = val_fraction_rate
        
        data0 = torch.load(self.data_dir+'/embedding_dict_data0.pth')
        data0 =torch.cat([data0,torch.zeros(len(data0),1)],1)
        data1 = torch.load(self.data_dir+'/embedding_dict_data1.pth')
        data1 = torch.cat([data1,torch.ones(len(data1),1)],1)
        data = torch.cat([data0,data1],0)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data = data[index,:]
        self.data = data.detach().numpy()
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        train_test_split = int(self.fraction_rate*len(self.data))
        insample = self.data[:train_test_split,:]
        test_data = self.data[train_test_split:,:]
        train_val_split = int((1-self.val_fraction_rate)*len(insample))
        train_data = insample[:train_val_split,:]
        val_data  = insample[train_val_split:,:]
        self.dataset_train = SklearnDataset(X=train_data[:,:-1],y = train_data[:,-1].astype(int))
        self.dataset_val = SklearnDataset(X=val_data[:,:-1],y = val_data[:,-1].astype(int))
        self.dataset_test = SklearnDataset(X=test_data[:,:-1],y = test_data[:,-1].astype(int))
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=gpu)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=gpu)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=gpu)

if __name__=='__main__':
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    early_stop_callback = EarlyStopping(monitor="val_loss",
     min_delta=0.00,
     patience=10,
     verbose=False,
     mode="min")
    
    trainer = pl.Trainer(accelerator='auto',
                         max_epochs=100,
                        callbacks=[checkpoint_callback,
                                   early_stop_callback,
                                   ],
                        logger=wandb_logger,
                        strategy="fsdp",
                        check_val_every_n_epoch=1,
                        devices=6,
                        precision=16,
                        )
    model = BaseLineModel(lr=0.01)
    data_module = MyDataModule(batch_size=256,num_workers=10)
    trainer.fit(model,data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()



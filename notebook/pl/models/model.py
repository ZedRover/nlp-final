import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm


class BaseLineModel(pl.LightningModule):

    def __init__(self,lr):
        super().__init__()
        self.l1 = torch.nn.Linear(300, 2,bias=False)
        self.accuracy = tm.Accuracy()
        self.save_hyperparameters()
        self.lr = lr
        self.calc_loss = torch.nn.CrossEntropyLoss()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.l1(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.calc_loss(self(x).squeeze(1), y.long())
        self.log('train_loss',loss)
        self.log('train_acc',self.accuracy(self(x).squeeze(1),y))
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.calc_loss(self(x).squeeze(1), y.long())

        self.log('val_loss', loss)
        self.log('val_loss',loss)
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.calc_loss(self(x).squeeze(1), y.long())
        self.log('test_loss', loss)
        self.log('test_acc',self.accuracy(self(x).squeeze(1),y))
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
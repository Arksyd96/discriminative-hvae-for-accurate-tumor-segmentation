import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
# for evaluation 
from torchmetrics import ConfusionMatrix
import pandas as pd
import seaborn as sn

class Classifier(pl.LightningModule):
    def __init__(
        self,
        backbone:       str = 'resnet18',
        in_channels:    int = 1,        
        pretrained:     bool = False,
        num_classes:    int = 10,
        lr:             float = 1e-4,
        weight_decay:   float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 'Backbone not supported'
        assert num_classes >= 2, 'Number of classes must be greater than or equal to 2'

        self.num_classes = num_classes
        self.model = models.__dict__[backbone](weights='DEFAULT' if pretrained else None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # for validation steps
        self.validation_step_outputs = []

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, _, y = batch
        x, y = x[:, 0, None, ...].to(dtype=torch.float32), y.to(dtype=torch.long) # picking image only

        y_hat = self(x)

        # compute loss
        loss = F.cross_entropy(F.log_softmax(y_hat, dim=1), F.one_hot(y, num_classes=self.num_classes).float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        x, y = x[:, 0, None, ...].to(dtype=torch.float32), y.to(dtype=torch.long) # picking image only

        y_hat = self(x)

        # compute loss
        loss = F.cross_entropy(F.log_softmax(y_hat, dim=1), F.one_hot(y, num_classes=self.num_classes).float())

        # get predictions
        y_hat = torch.argmax(y_hat, dim=1)
        self.validation_step_outputs.append((y_hat.cpu(), y.cpu(), loss.item()))

    def on_validation_epoch_end(self) -> None:
        eval_preds = torch.cat([output[0] for output in self.validation_step_outputs])
        eval_targs = torch.cat([output[1] for output in self.validation_step_outputs])
        mean_loss = np.array([output[2] for output in self.validation_step_outputs]).mean()

        confusion_matrix = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')
        matrix = confusion_matrix(eval_preds, eval_targs)

        accuracy = matrix.trace() / matrix.sum()
        precision = np.array([matrix[i, i] / matrix.sum(axis=0)[i] for i in range(self.num_classes)])
        recall = np.array([matrix[i, i] / matrix.sum(axis=1)[i] for i in range(self.num_classes)]) 
        f1_score = 2 * precision * recall / (precision + recall)

        self.log_dict({
            'val_loss': mean_loss,
            'val_accuracy': accuracy.mean(),
            'val_precision': precision.mean(),
            'val_recall': recall.mean(),
            'val_f1_score': f1_score.mean()
        }, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()
        
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
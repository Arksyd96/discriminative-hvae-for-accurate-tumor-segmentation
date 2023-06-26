import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from functools import partial

from .base import (
    TimePositionalEmbedding, EncodingBlock, DecodingBlock
)

class ResUNet(pl.LightningModule):
    def __init__(
        self, 
        input_shape,
        num_classes,
        timesteps,
        use_classifier  = False,
        cls_classes     = None,
        lr              = 1e-4,
        weight_decay    = 1e-6,
        **kwargs
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.num_classes = num_classes
        self.use_classifier = use_classifier
        self.cls_classes = cls_classes
        self.latent_shape = np.array(input_shape[1:]) // 2 ** 4
        assert self.latent_shape[0] > 1, 'Input shape is too small'
        assert cls_classes is not None if use_classifier else True, 'Number of classes must be specified'

        self.in_conv = nn.Conv2d(self.in_channels, 128, kernel_size=3, padding='same')
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=128, T=timesteps),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.encoder = nn.ModuleList([
            EncodingBlock(in_channels=128, out_channels=128, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=128, out_channels=256, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=256, temb_dim=128 * 4, downsample=True, attn=True, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=512, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.bottleneck = EncodingBlock(in_channels=512, out_channels=512, temb_dim=128 * 4, downsample=False, attn=True, num_blocks=2, groups=32)

        self.decoder = nn.ModuleList([
            DecodingBlock(in_channels=512 + 512, out_channels=512, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=512 + 256, out_channels=256, temb_dim=128 * 4, upsample=True, attn=True, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 256, out_channels=256, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 128, out_channels=128, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32)
        ])

        out_channels = self.num_classes if self.num_classes > 2 else 1
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        )

        if self.use_classifier:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * np.prod(self.latent_shape), 512),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, cls_classes if cls_classes > 2 else 1)
            )

        self.save_hyperparameters()

    def forward(self, x, time):
        assert x.shape[0] == time.shape[0], 'Batch size of x and time must be the same'
        temb = self.positional_encoder(time)
        skip_connections = []

        x = self.in_conv(x)
        skip_connections.append(x)
        
        # encoding part
        for block in self.encoder:
            x = block(x, temb)
            skip_connections.append(x)

        # bottleneck
        x = self.bottleneck(x, temb)
        if self.use_classifier:
            cls = self.classifier(x)

        # decoding part
        for block in self.decoder:
            x = block(torch.cat([x, skip_connections.pop()], dim=1), temb)

        x = torch.cat([x, skip_connections.pop()], dim=1)
        assert len(skip_connections) == 0, 'Skip connections must be empty'

        if self.use_classifier:
            return self.out_conv(x), cls
        return self.out_conv(x)
    
    def on_train_start(self) -> None:
        self.positional_encoder[0].embedding = self.positional_encoder[0].embedding.to(self.device)
    
    def training_step(self, batch, batch_idx):
        x, time, cls = batch
        input, mask = x[:, 0, None, ...].to(dtype=torch.float32), x[:, 1, None, ...].to(dtype=torch.float32) # TODO: FLOAT16
        time, cls = time.to(dtype=torch.long), cls.to(dtype=torch.long)

        mask_hat, cls_hat = self.forward(input, time)
        loss, log = self.compute_losses(mask, mask_hat, cls, cls_hat)

        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, time, cls = batch
        input, mask = x[:, 0, None, ...].to(dtype=torch.float32), x[:, 1, None, ...].to(dtype=torch.float32)
        time, cls = time.to(dtype=torch.long), cls.to(dtype=torch.long)

        mask_hat, cls_hat = self.forward(input, time)
        loss, log = self.compute_losses(mask, mask_hat, cls, cls_hat, suffix='val')

        # compute accuracy
        if self.use_classifier:
            acc = (torch.argmax(cls_hat, dim=1) == cls).sum() / cls.shape[0]
            log['val_acc'] = acc

        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True)

    def compute_losses(self, y, y_hat, cls, cls_hat, suffix=''):
        # dice loss
        if self.num_classes > 2:
            dice_loss = self.dice_loss(
                F.softmax(y_hat, dim=1), 
                F.one_hot(y.to(torch.int32), num_classes=self.num_classes).permute(0, 3, 1, 2).to(torch.float32)
            )
        else: # binary dice loss
            dice_loss = self.dice_loss(
                torch.sigmoid(y_hat), 
                y.to(torch.float32)
            )

        # classification loss
        cls_loss = 0
        if self.use_classifier:
            if self.cls_classes > 2:
                cls_loss = F.cross_entropy(F.softmax(cls_hat, dim=1), F.one_hot(cls, num_classes=self.cls_classes))
            else: # binary cross entropy
                cls_loss = F.binary_cross_entropy_with_logits(cls_hat.squeeze(), cls.float())

        log = {
            '{}loss'.format(suffix): dice_loss + cls_loss,
            '{}dice_loss'.format(suffix): dice_loss,
            '{}cls_loss'.format(suffix): cls_loss
        }

        return dice_loss + cls_loss, log

    def dice_loss(self, input, target, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        input, target = input.flatten(0, 1), target.flatten(0, 1)

        inter = 2 * (input * target).sum(dim=(1, 2))
        sets_sum = input.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return 1 - dice.mean()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
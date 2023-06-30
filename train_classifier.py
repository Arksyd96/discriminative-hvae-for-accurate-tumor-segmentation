import numpy as np
import torch
import pytorch_lightning as pl
import os
from omegaconf import OmegaConf
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.preprocessing import BRATSDataModule
from modules.classifier import Classifier

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if debugging:
        torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    global_seed(42)
    torch.set_float32_matmul_precision('high')
    
    # logger
    logger = wandb_logger.WandbLogger(
        project='discrimnative-hvae-for-accurate-tumor-segmentation', 
        name='Training classifier'
    )

    # model
    model = Classifier(
        backbone    = 'resnet34',
        pretrained  = False,
        num_classes = 2
    )
    
    # data module
    datamodule = BRATSDataModule(
        target_shape = (64, 256, 256),
        n_samples   = 1000,
        train_ratio = 0.85,
        modalities  = ['flair', 'seg'],
        binarize    = True,
        balance     = True,
        batch_size  = 32,
        shuffle     = True,
        num_workers = 6
    )
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath     =  './checkpoints',
        save_top_k  = 1,
        every_n_epochs = 25,
        filename    = 'binary_classifier-{epoch}',
    )
    
    # training
    trainer = pl.Trainer(
        logger      = logger,
        # strategy  = "ddp",
        # devices   = 4,
        # num_nodes = 2,
        accelerator = 'gpu',
        precision   = 32,
        max_epochs  = 500,
        log_every_n_steps   = 1,
        enable_progress_bar = True,
        callbacks   = [checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    
        
    

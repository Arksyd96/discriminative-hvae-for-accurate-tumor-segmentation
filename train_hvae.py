import numpy as np
import torch
import pytorch_lightning as pl
import os
from omegaconf import OmegaConf
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.preprocessing import BRATSDataModule
from modules.loggers import ImageSampler
from modules.autoencoders import HamiltonianAutoencoder

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
    
    # loading config file
    CONFIG_PATH = './config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    cfg = OmegaConf.load(CONFIG_PATH)
    
    # logger
    logger = wandb_logger.WandbLogger(
        project='discrimnative-hvae-for-accurate-tumor-segmentation', 
        name='Training HVAE',
        # id='24hyhi7b',    # in case we want to resume
        # resume="must"
    )

    # model
    model = HamiltonianAutoencoder(**cfg.autoencoder)
    
    # data module
    datamodule = BRATSDataModule(**cfg.data)
        
    image_sampler_logger = ImageSampler(
        n_samples=5, 
        every_n_epochs=50,
        label='Image sampling',
    )
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(
        **cfg.callbacks.checkpoint,
        filename='ckpt-{epoch}',
    )
    
    # training
    trainer = pl.Trainer(
        logger=logger,
        strategy="ddp_find_unused_parameters_true",
        devices=4,
        num_nodes=2,
        accelerator='gpu',
        precision=32,
        max_epochs=5000,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, image_sampler_logger]
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    
        
    

import torch
import pytorch_lightning as pl
import wandb
import numpy as np

class ImageReconstructionLogger(pl.Callback):
    def __init__(self,
        modalities=['t1', 't1ce', 't2', 'flair'], 
        n_samples=5,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.modalities = modalities

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # sample images
            pl_module.eval()
            rand_idx = np.random.randint(0, trainer.val_dataloaders.dataset.__len__(), size=(self.n_samples,))
            x, pos = trainer.val_dataloaders.dataset[rand_idx]
            x, pos = x.to(pl_module.device, torch.float32), pos.to(pl_module.device, torch.long)
            x, pos = x[:self.n_samples], pos[:self.n_samples]

            with torch.set_grad_enabled(True):
                x_hat = pl_module(x, pos)['x_hat']

            originals = torch.cat([
                torch.hstack([img for img in x[:, idx, ...]])
                for idx in range(self.modalities.__len__())
            ], dim=0)
            
            reconstructed = torch.cat([
                torch.hstack([img for img in x_hat[:, idx, ...]])
                for idx in range(self.modalities.__len__())
            ], dim=0)
            
            img = torch.cat([originals, reconstructed], dim=0)
            
            wandb.log({
                'Reconstruction examples': wandb.Image(
                    img.detach().cpu().numpy(), 
                    caption='{} - {} (Top are originals)'.format(self.modalities, trainer.current_epoch)
                )
            })
            
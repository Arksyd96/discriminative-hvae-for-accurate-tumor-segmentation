import numpy as np
import torch
import pytorch_lightning as pl
import wandb

class ImageSampler(pl.Callback):
    def __init__(self, 
        n_samples=5,
        every_n_epochs=1,
        label='Sampling'
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.label = label
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                # sample images
                pl_module.eval()
                generated = pl_module.sample_img(n_samples=self.n_samples, device=pl_module.device)
                
                # channel wise grid
                img_grid = torch.cat([ 
                    torch.hstack([img for img in generated[:, idx, ...]])
                    for idx in range(generated.shape[1])
                ], dim=0)

                img_grid = img_grid.unsqueeze(-1).detach().cpu().numpy()
                img_grid = (img_grid * 255).astype(np.uint8) # denormalize
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        img_grid, caption='{}'.format(self.label)
                    )
                })
            
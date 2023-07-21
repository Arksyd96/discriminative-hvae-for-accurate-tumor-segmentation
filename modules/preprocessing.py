import numpy as np
import torch
import pytorch_lightning as pl
from nibabel import load
from nibabel.processing import resample_to_output
from tqdm import tqdm
import os

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

class BRATSDataModule(pl.LightningDataModule):
    def __init__(self,
        target_shape    = (64, 128, 128),
        n_samples       = 500,
        modalities      = ['t1', 't1ce', 't2', 'flair', 'seg'],
        binarize        = True,
        balance         = True,
        npy_path        = './data/brats_preprocessed.npy',
        root_path       = '../common_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021',
        batch_size      = 32,
        shuffle         = True,
        num_workers     = 4,
        **kwargs
    ) -> None:
        assert all([m in ['t1', 't1ce', 't2', 'flair', 'seg'] for m in modalities]), 'Invalid modality!'
        
        super().__init__()
        self.num_modalities = len(modalities)
        self.prepare_data_per_node = True

        # just for a faster access
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.npy_path):
            print('Loading dataset from NiFTI files...')
            placeholder = np.zeros(shape=(
                self.hparams.n_samples, 
                self.num_modalities, 
                self.hparams.target_shape[1], 
                self.hparams.target_shape[2], 
                self.hparams.target_shape[0]
            ))

            for idx, instance in enumerate(tqdm(os.listdir(self.hparams.root_path)[: self.hparams.n_samples], position=0, leave=True)):
                # loading models
                volumes = {}
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = load(os.path.join(self.hparams.root_path, instance, instance + f'_{m}.nii.gz'))

                # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
                orig_shape = volumes[self.hparams.modalities[0]].shape
                scale_factor = (orig_shape[0] / self.hparams.target_shape[1], # height
                                orig_shape[1] / self.hparams.target_shape[2], # width
                                orig_shape[2] / self.hparams.target_shape[0]) # depth

                # Resample the image using trilinear interpolation
                # Drop the last extra rows/columns/slices to get the exact desired output size
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=1).get_fdata()
                    volumes[m] = volumes[m][:self.hparams.target_shape[1], :self.hparams.target_shape[2], :self.hparams.target_shape[0]]

                # binarizing the mask (for simplicity), you can comment out this to keep all labels
                if self.hparams.binarize and 'seg' in self.hparams.modalities:
                    volumes['seg'] = (volumes['seg'] > 0).astype(np.float32)

                # saving models
                for idx_m, m in enumerate(self.hparams.modalities):
                    placeholder[idx, idx_m, :, :] = volumes[m]

            print('Saving dataset as npy file...')    
            # saving the dataset as a npy file
            np.save(self.hparams.npy_path, placeholder)
                
            print('Saved!')
            print('Max: {}, Min: {}'.format(placeholder.max(), placeholder.min()))
                
        else:
            print('Dataset already exists at {}'.format(self.hparams.npy_path))
        
    def setup(self, stage='fit'):
        print('Loading dataset from npy file...')
        data = torch.from_numpy(np.load(self.hparams.npy_path))
        data = data.permute(0, 4, 1, 2, 3) # depth first

        # switching to 2D
        D, W, H = self.hparams.target_shape
        data = data.reshape(data.shape[0] * D, -1, W, H)

        # normalize the data [0, 1] example by example (only images)
        for idx in tqdm(range(data.shape[0])):
            if data[idx, 0].max() != 0:
                data[idx, 0] = (data[idx, 0] - data[idx, 0].min()) / (data[idx, 0].max() - data[idx, 0].min())

        # removing empty slices (no tumors)
        data = data[data[:, 1].sum(dim=(1, 2)) != 0]

        # balancing the dataset according to the size of the tumors
        max_tumor_size = data[:, 1].sum(dim=(1, 2)).max()
        bins = np.arange(200, max_tumor_size, 25) # 25 is the bin size
        slices_per_level = self.hparams.n_samples * D // len(bins) 
        print('Max tumor size: {}, Slices per level: {}, Bins: {}'.format(max_tumor_size, slices_per_level, len(bins)))

        for idx in tqdm(range(bins.__len__() - 1)):
            curr_bin = data[(data[:, 1].sum(dim=(1, 2)) >= bins[idx]) & (data[:, 1].sum(dim=(1, 2)) < bins[idx + 1])]
            curr_bin = curr_bin[:slices_per_level]

            if idx == 0:
                data_balanced = curr_bin
            else:
                data_balanced = torch.cat([data_balanced, curr_bin], dim=0)

        # # fill the empty space with random slices that have at least a tumor size of 400
        # if data_balanced.shape[0] < self.hparams.n_samples * D:
        #     n_samples = self.hparams.n_samples * D - data_balanced.shape[0]
        #     random_samples = data[(data[:, 1].sum(dim=(1, 2)) >= 400)]
        #     random_samples = random_samples[torch.randperm(random_samples.shape[0])][:n_samples]
        #     data_balanced = torch.cat([data_balanced, random_samples], dim=0)

        print('Train shape:', data_balanced.shape) 
        print('Min: {}, Max: {}'.format(data_balanced.min(), data_balanced.max()))
        
        self.train_dataset = IdentityDataset(data_balanced)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=self.hparams.shuffle, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )
    
    
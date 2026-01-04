from torch.utils.data import DataLoader
import torch
from . import classification
from . import regression
import lightning as L
from . import ldl

class Data(L.LightningDataModule):
    def __init__(self, dataset: str, batch_size: int, test_batch_size: int, task=None, split_num=0., task_type=None,
                 val_perc=0.01, nw=8, data_aug=True, aug_type='basic', root='.data', fold_idx=-1,**kwargs):
        super().__init__()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.task = task
        self.val_perc = val_perc
        self.split_num = split_num
        self.task_type = task_type
        self.data_aug = data_aug
        self.aug_type = aug_type
        self.fold_idx = fold_idx
        ldl_list = ["SBU_3DFE", "Scene", "Gene", "Movie", "RAF_ML", "Ren_Cecps", 
                    "SJAFFE", "M2B", "SCUT_FBP5500", "Twitter_LDL", "Flickr_LDL", "SCUT_FBP"]
        if self.dataset_name in ['cifar10', 'cifar100', 'mnist', 'svhn']:
            self.train_dataset, self.eval_dataset, self.test_dataset = classification.get_datasets(
                root=root, data_aug=self.data_aug, name=self.dataset_name, perc=self.val_perc, aug_type=aug_type)
            self.train_loader, self.val_loader, self.test_loader = classification.get_dataloaders(
                self.train_dataset, self.eval_dataset, self.test_dataset, batch_size=self.batch_size, test_batch_size=self.test_batch_size, nw=nw)
        elif self.dataset_name in ldl_list:
            run_idx = int(self.split_num) if self.split_num is not None else 0
            self.train_dataset, self.val_dataset, self.test_dataset = ldl.get_ldl_datasets(
                self.dataset_name, run_idx=run_idx, fold_idx=self.fold_idx
            )
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=nw)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=nw)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=nw)
        elif self.dataset_name == 'uci':
            self.train_dataset, self.val_dataset, self.test_dataset = regression.get_UCI_datasets(
                root+'/UCI_Datasets', self.task, self.split_num, keys=('train', 'val', 'test')
            )

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                         pin_memory=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                          pin_memory=False)

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return [self.test_loader, self.val_loader]

    def test_dataloader(self) -> DataLoader:
        return self.test_loader

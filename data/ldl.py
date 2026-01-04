import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold 
class LDL_Feature_Dataset(Dataset):
    def __init__(self, dataset_name, run_idx=0, mode='train'):
        # 路径指向 ../Data/feature
        root_dir = os.path.join('../Data/feature', dataset_name, f'run_{run_idx}')
        
        if not os.path.exists(root_dir):
             # 尝试 fallback 到 image 目录（有的数据集可能放在那里但已经是特征了）
             alt_dir = os.path.join('../Data/image', dataset_name, f'run_{run_idx}')
             if os.path.exists(alt_dir):
                 root_dir = alt_dir
             else:
                 raise FileNotFoundError(f"Data not found at {root_dir}")

        feature_path = os.path.join(root_dir, f'{mode}_feature.npy')
        label_path = os.path.join(root_dir, f'{mode}_label.npy')
        
        # 直接读取，不做任何处理
        self.features = np.load(feature_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32)
        
        self.features = torch.from_numpy(self.features)
        self.labels = torch.from_numpy(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

def get_ldl_datasets(dataset_name, run_idx=0, fold_idx=-1):
    """
    fold_idx: 
      -1:  【正常模式】全量训练集 + 真实测试集 (用于最终评估)
      0-4: 【搜索模式】只加载训练集，并在内部做 5 折切分 
    """
    # 1. 先加载 Run_X 的全量训练集
    full_train_set = LDL_Feature_Dataset(dataset_name, run_idx, mode='train')
    
    # === 搜索模式：使用 KFold 进行标准切分 ===
    if fold_idx >= 0:
        # 定义 KFold：5折，打乱，固定种子42
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 获取所有的切分方案 (得到一个列表，里面有 5 个 (train_idx, val_idx) 元组)
        all_splits = list(kf.split(range(len(full_train_set))))
        
        # 取出当前 fold_idx 对应的那一组索引
        train_indices, val_indices = all_splits[fold_idx]
        
        # 构建 Subset
        train_d = Subset(full_train_set, train_indices)
        val_d = Subset(full_train_set, val_indices)
        
        # 在搜索阶段，用切出来的验证集作为 test_loader
        return train_d, val_d, val_d

    else:
        test_set = LDL_Feature_Dataset(dataset_name, run_idx, mode='test')
        return full_train_set, test_set, test_set
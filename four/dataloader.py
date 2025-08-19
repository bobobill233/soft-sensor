import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class TensorDataset(Dataset):
    def __init__(self, tensor_files):
        self.tensor_files = tensor_files

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        file_info = list(self.tensor_files.values())[idx]
        gasf_tensor = torch.load(file_info['gasf_tensor_path'])
        mtf_tensor = torch.load(file_info['mtf_tensor_path'])
        label = torch.tensor(float(file_info['label']), dtype=torch.float)
        return gasf_tensor, mtf_tensor, label

class DataLoaderWrapper:
    def __init__(self, source_folder, train_csv, val_csv):
        self.source_folder = source_folder
        self.train_csv_path = os.path.join(source_folder, train_csv)
        self.val_csv_path = os.path.join(source_folder, val_csv)

        # Load CSV data
        self.train_data = pd.read_csv(self.train_csv_path)
        self.val_data = pd.read_csv(self.val_csv_path)

        # Normalize the labels
        self.scaler = MinMaxScaler()
        self.train_data['label'] = self.scaler.fit_transform(self.train_data[['label']])
        self.val_data['label'] = self.scaler.transform(self.val_data[['label']])

        # Create dictionaries for training and validation datasets
        self.train_tensor_dict = self._create_tensor_dict(self.train_data)
        self.val_tensor_dict = self._create_tensor_dict(self.val_data)

        # Create datasets
        self.train_dataset = TensorDataset(self.train_tensor_dict)
        self.val_dataset = TensorDataset(self.val_tensor_dict)

    def _create_tensor_dict(self, df):
        tensor_dict = {}
        for idx, row in df.iterrows():
            tensor_dict[idx] = {
                'gasf_tensor_path': row['gasf_tensor_path'],
                'mtf_tensor_path': row['mtf_tensor_path'],
                'label': row['label']
            }
        return tensor_dict

    def get_loaders(self, train_batch_size, val_batch_size):
        train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=False)
        valid_loader = DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=True)
        return train_loader, valid_loader

    def get_scaler(self):
        return self.scaler

# # 示例使用
# source_folder = 'D:\\flow_idea2\\tensormixgasf3d'
# train_csv = 'train_data.csv'
# val_csv = 'valid_data.csv'
#
# # 创建DataLoaderWrapper实例
# data_loader_wrapper = DataLoaderWrapper(source_folder, train_csv, val_csv)
#
# # 获取训练和验证数据加载器，设置不同的 batch size
# train_loader, valid_loader = data_loader_wrapper.get_loaders(train_batch_size=4, val_batch_size=6)
#
# # 使用验证数据加载器
# for gasf_tensor, mtf_tensor, label in valid_loader:
#     print(gasf_tensor.size(), mtf_tensor.size(), label)

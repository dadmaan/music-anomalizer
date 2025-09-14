from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class DatasetSampler(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].astype('float32')
    
    
class DatasetSamplerInt(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].astype('long')
    
class DataHandler:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None
        self.num_features = None
        self.num_data = None

    def load_data(self):
        # Convert tuple to list if necessary for shuffling
        if isinstance(self.data, tuple):
            self.data = list(self.data)
        random.shuffle(self.data)
        
        self.num_data = len(self.data)
        num_train = int(self.num_data * 0.8)
        
        train_data = self.data[:num_train]
        val_data = self.data[num_train:]
        print('The number of train:', len(train_data))
        print('The number of validation:', len(val_data))

        # Get num_features from the first data item since self.data might be a list now
        if hasattr(self.data, 'shape'):
            self.num_features = self.data.shape[1]
        else:
            # If data is a list, get features from first element
            if len(self.data) > 0 and self.data[0] is not None:
                self.num_features = self.data[0].shape[0] if hasattr(self.data[0], 'shape') else len(self.data[0])
            else:
                raise ValueError("Cannot determine number of features: data is empty or contains None values")
        
        train_params = {'batch_size': self.batch_size, 'shuffle': True, 'pin_memory': True, 'num_workers': 4}
        val_params = {'batch_size': self.batch_size, 'shuffle': False, 'pin_memory': True, 'num_workers': 4}
        
        self.train_set = DataLoader(DatasetSampler(train_data), **train_params)
        self.val_set = DataLoader(DatasetSampler(val_data), **val_params)

    def get_train_set(self):
        return self.train_set

    def get_val_set(self):
        return self.val_set

    def get_num_features(self):
        return self.num_features

    def get_num_data(self):
        return self.num_data
    
    def get_num_train(self):
        return self.num_data

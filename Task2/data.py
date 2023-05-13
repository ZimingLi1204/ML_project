from torch.utils.data import Dataset
import os
import torch

class Mydataset(Dataset):

    def __init__(self, mode: torch.tensor, data : torch.tensor , label: 'None|torch.tensor'):
        '''
        mode: train/val/test
        data: N * k * 192 * 192.
        label: 每个data的groundtruth mask, N * k * 192 * 192
        '''
        self.mode = mode
        self.data = data
        self.label = label
        ###这里还要给每个data赋予一个标记, 来记录这个2d数据是属于哪一个3d数据的.
        self.numberic_label = torch.arange(data.shape[0]).repeat(data.shape[1], 1)

        #所有data reshape成N' * 192 * 192
        self.data = self.data.reshape(-1, data.shape[2], data.shape[3])
        if mode != 'test':
            self.label = self.label.reshape(-1, label.shape[2], label.shape[3])

    def __len__(self):
        
        return self.data.shape[0]
   
    def __getitem__(self, index):

        img = self.data[index]

        if self.mode == 'test':
            mask = None
        else:
            mask = self.label[index]
            promt, promt_type = get_promt(mask, img)

        return img, mask, promt, promt_type
        

def load_data_train(cfg):
    train_data_dir = os.path.join(cfg["data"]["data_root"], "train_img")
    train_label_dir = os.path.join(cfg["data"]["data_root"], "train_label")
    
    train_dataset, val_dataset = load_train_data_from_dir(train_data_dir, train_label_dir)
    
    return train_dataset, val_dataset
    

def load_data_test(cfg):
    test_data_dir = os.path.join(cfg["data"]["data_root"], "test_img")
    test_dataset = load_test_data_from_dir(test_data_dir)


def get_promt(img, mask):
    ###TODO###
    #根据输入img和mask生成promt
    promt = None
    promt_type = None # box or mask or points !!!
    return promt, promt_type

def load_train_data_from_dir(data_dir, label_dir) -> tuple(Mydataset, Mydataset):
    ###TODO###
    #根据路径提取并处理数据, 划分训练/验证集. 这部分数据都是有label的

    train_data = None
    val_data = None
    train_label = None
    val_label = None

    mydataset_train = Mydataset(mode='train',data=train_data, label = train_label)
    mydataset_val = Mydataset(mode='train', data=val_data, label = val_label)

    return mydataset_train, mydataset_val


def load_test_data_from_dir(data_dir) -> Mydataset:
    ###TODO###
    #根据路径提取并处理数据, 生成测试集, 无label

    test_data = None
    test_label = None #fix, 不用修改

    mydataset_test = Mydataset(mode='test', data=test_data, label=test_label)

    return mydataset_test
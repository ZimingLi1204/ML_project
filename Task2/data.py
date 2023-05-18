from torch.utils.data import Dataset
import os
import torch

class Mydataset(Dataset):

    def __init__(self, mode: torch.tensor, data : torch.tensor , label: 'None|torch.tensor', index: torch.tensor):
        '''
        mode: train/val/test
        data: N * imgsize * imgsize.
        label: 每个data的groundtruth mask, N * imgsize * imgsize
        '''
        self.mode = mode
        self.data = data
        self.label = label
        self.index = index
        assert (self.data.shape[0] == self.label.shape[0])
        assert (self.data.shape[0] == self.index.shape[0])

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
    train_index = None ###每个2d图片对应的3d图片的index
    val_index = None ###每个2d图片对应的3d图片的index

    mydataset_train = Mydataset(mode='train',data=train_data, label = train_label, index=train_index)
    mydataset_val = Mydataset(mode='train', data=val_data, label = val_label, index = val_index)

    return mydataset_train, mydataset_val


def load_test_data_from_dir(data_dir) -> Mydataset:

    ###TODO###
    #根据路径提取并处理数据, 生成测试集, 无label

    test_data = None
    test_label = None #fix, 不用修改
    test_index = None ###每个2d图片对应的3d图片的index

    mydataset_test = Mydataset(mode='test', data=test_data, label=test_label, index=test_index)

    return mydataset_test
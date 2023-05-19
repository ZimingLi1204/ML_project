from torch.utils.data import Dataset
import os
import torch
import numpy as np
import scipy.io as scio
import sys
import pdb
sys.path.append("..")
class Mydataset(Dataset):

    def __init__(self, mode: torch.tensor, img : torch.tensor , mask: 'None|torch.tensor', name: list, slice_id: list, category: list, promt_type="single_point"):
        '''
        mode: train/val/test
        data: N * 512 * 512.
        mask: 每个data的groundtruth 0-1mask, N * 512 * 512
        name: 每个data对应的CT编号 1-10, 21-40   list中每个元素格式为 str(00xx) 如 "0001"      N * 1
        slice_id: 每个data对应的切片编号 每个CT有80-150个切片    list中每个元素格式为 str(xxx)    N * 1
        category: 每个mask对应的类别 范围为1-13     N * 1
        '''
        self.mode = mode
        self.img = img
        self.mask = mask
        self.name = name
        self.slice_id = slice_id
        self.category = category
        self.promt_type = promt_type

    def __len__(self):
        
        return self.img.shape[0]
   
    def __getitem__(self, index):

        img = self.img[index]
        promt, promt_label = None, None
        promt_type = self.promt_type

        mask = self.mask[index]
        promt = get_promt(img, mask, promt_type)
        if isinstance(promt, tuple):
            promt, promt_label = promt

        return img, mask, promt, promt_label, promt_type
        

def load_data_train(cfg):
    data_train_path = os.path.join(cfg['data']['data_root'], "pre_processed_dataset1_train")
    data_val_path = os.path.join(cfg['data']['data_root'], "pre_processed_dataset1_val")
    # data_train_path = "BTCV/pre_processed_dataset1_train"
    # data_val_path = "BTCV/pre_processed_dataset1_val"
    train_dataset, val_dataset = load_train_data_from_dir(data_train_path, data_val_path)
    
    return train_dataset, val_dataset  # type = Mydataset
    

def load_data_test(cfg):
    # test_data_dir = os.path.join(cfg["data"]["data_root"], "test_img")
    # test_dataset = load_test_data_from_dir(test_data_dir)
    data_test_path = os.path.join(cfg['data']['data_root'], "pre_processed_dataset1_test")
    # data_test_path = "BTCV/pre_processed_dataset1_test"
    test_dataset = load_test_data_from_dir(data_test_path)

    return test_dataset


def get_promt(img, mask, promt_type = "single_point", point_num = 1, box_num = 1):
    ###TODO###
    #根据输入img和mask生成promt
    # box or mask or points or single_point!!!
    # 需要保证生成的 point promt均在mask前景中
    # 不同类型promt 的具体格式见 segment_anything/predictor.py 104-130行注释

    promt = None

    if promt_type == "single_point":   # 单点 1个XY坐标 和 1个01 label
        coord = np.random.randint(low=1, high=512, size=(1, 2))
        while mask[coord[0, 0], coord[0, 1]] == 0:      # 随机取一个在mask前景中的XY坐标
            coord = np.random.randint(low=1, high=512, size=(1, 2))
        label = np.array([mask[coord[0, 0], coord[0, 1]]])
        promt = coord, label
    elif promt_type == "points":   # 多点   N个XY坐标 和 N个01 label
        coord = np.random.randint(low=1, high=512, size=(point_num, 2))
        label = np.array([mask[coord[0, 0], coord[0, 1]] for i in range(point_num)])
        promt = coord, label
    elif promt_type == "box":   # 边界框  形如XYXY
        coord = np.random.randint(low=1, high=512, size=4)
        promt = coord
    elif promt_type == "mask":   # mask类型prompt
        pass
    else:
        raise Exception

    return promt

def load_train_data_from_dir(data_train_path, data_val_path):
    #根据路径提取并处理数据, 划分训练/验证集. 这部分数据都是有label的

    print("loading img & mask......")
    train_data = np.load(data_train_path+'.npz')
    val_data = np.load(data_val_path+'.npz')

    print("loading name & slice_id & category......")
    train_info = scio.loadmat(data_train_path+'.mat')
    val_info = scio.loadmat(data_val_path+'.mat')

    img_train = train_data["img"]
    mask_train = train_data["mask"]
    img_val = val_data["img"]
    mask_val = val_data["mask"]
    name_train = train_info["name"]
    name_val = val_info["name"]
    slice_id_train = train_info["slice_id"]
    slice_id_val = val_info["slice_id"]
    category_train = train_info["category"]
    category_val = val_info["category"]
    # print(mask_train.max())

    mydataset_train = Mydataset(mode='train',img=img_train, mask=mask_train, name=name_train, slice_id=slice_id_train, category=category_train)
    mydataset_val = Mydataset(mode='train', img=img_val, mask=mask_val, name=name_val, slice_id=slice_id_val, category=category_val)

    return mydataset_train, mydataset_val


def load_test_data_from_dir(data_test_path) -> Mydataset:
    #根据路径提取并处理数据, 生成测试集, 有label

    print("loading test img & mask......")
    test_data = np.load(data_test_path + '.npz')

    print("loading test name & slice_id & category......")
    test_info = scio.loadmat(data_test_path + '.mat')

    img_test = test_data["img"]
    mask_test = test_data["mask"]
    name_test = test_info["name"]
    slice_id_test = test_info["slice_id"]
    category_test = test_info["category"]

    mydataset_test = Mydataset(mode='test', img=img_test, mask=mask_test, name=name_test, slice_id=slice_id_test,
                                category=category_test)


    return mydataset_test
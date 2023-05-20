from torch.utils.data import Dataset
import os
import torch
import numpy as np
import scipy.io as scio
import sys
import pdb
from tqdm import tqdm
sys.path.append("..")
from utils import get_promt

class Mydataset(Dataset):

    def __init__(self, mode: torch.tensor, img : torch.tensor , mask: torch.tensor, name: list, slice_id: list, category: list, promt_type="single_point", load_from_disk=False):
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
        self.load_from_disk = load_from_disk

    def __len__(self):
        
        return self.img.shape[0]
   
    def __getitem__(self, index):

        img = self.img[index]
        # if self.load_from_disk:
        #     img = img.copy()
        promt, promt_label = None, None
        promt_type = self.promt_type

        mask = self.mask[index]
        promt = get_promt(img, mask, promt_type)
        if isinstance(promt, tuple):
            promt, promt_label = promt

        return img, mask, promt, promt_label, promt_type
        

def load_data_train(cfg):

    data_train_path = os.path.join(cfg['data']['data_root'], cfg["data"]["data_name"] + '_' + "train")
    data_val_path = os.path.join(cfg['data']['data_root'],  cfg["data"]["data_name"] + '_' + "val")
    info_train_path = os.path.join(cfg['data']['data_root'], cfg["data"]["info_name"] + '_' + "train")
    info_val_path = os.path.join(cfg['data']['data_root'],  cfg["data"]["info_name"] + '_' + "val")
    
    train_dataset, val_dataset = load_train_data_from_dir(data_train_path, data_val_path, info_train_path, info_val_path, use_embedded=cfg['data']['use_embedded'], load_from_disk = cfg['data']['load_from_disk'])
    
    return train_dataset, val_dataset 
    

def load_data_test(cfg):

    data_test_path = os.path.join(cfg['data']['data_root'],  cfg["data"]["data_name"] + '_' + "test")
    info_test_path = os.path.join(cfg['data']['data_root'],  cfg["data"]["info_name"] + '_' + "test")
   
    test_dataset = load_test_data_from_dir(data_test_path, info_test_path, use_embedded=cfg['data']['use_embedded'], load_from_disk = cfg['data']['load_from_disk'])

    return test_dataset

def load_train_data_from_dir(data_train_path, data_val_path, info_train_path, info_val_path, use_embedded=False, load_from_disk = False):
    #根据路径提取并处理数据, 划分训练/验证集. 这部分数据都是有label的

    print("loading img & mask......")
    train_data = np.load(info_train_path+'.npz')
    val_data = np.load(info_val_path+'.npz')
    
    if use_embedded:
        if load_from_disk:
            train_embedded_data = np.load(data_train_path+'.npy', mmap_mode='r')
            val_embedded_data = np.load(data_val_path+'.npy', mmap_mode='r')
        else:
            #这里load data大约需要10分钟
            train_embedded_data = np.load(data_train_path+'.npy')
            val_embedded_data = np.load(data_val_path+'.npy')

    print("loading name & slice_id & category......")
    train_info = scio.loadmat(info_train_path+'.mat')
    val_info = scio.loadmat(info_val_path+'.mat')

    if use_embedded:
        img_train = train_embedded_data
        img_val = val_embedded_data
    else:
        img_train = train_data["img"]
        img_val = val_data["img"]

    mask_train = train_data["mask"]
    mask_val = val_data["mask"]
    name_train = train_info["name"]
    name_val = val_info["name"]
    slice_id_train = train_info["slice_id"]
    slice_id_val = val_info["slice_id"]
    category_train = train_info["category"]
    category_val = val_info["category"]
   
    mydataset_train = Mydataset(mode='train',img=img_train, mask=mask_train, name=name_train, slice_id=slice_id_train, category=category_train, load_from_disk=load_from_disk)
    mydataset_val = Mydataset(mode='train', img=img_val, mask=mask_val, name=name_val, slice_id=slice_id_val, category=category_val, load_from_disk=load_from_disk)

    return mydataset_train, mydataset_val

def load_test_data_from_dir(data_test_path, info_test_path, use_embedded=False, load_from_disk = False) -> Mydataset:
    #根据路径提取并处理数据, 生成测试集, 有label

    print("loading test img & mask......")
    test_data = np.load(info_test_path + '.npz')
    if use_embedded:
        if load_from_disk:
            test_embedded_data = np.load(data_test_path+'.npy', mmap_mode='r')
        else:
            test_embedded_data = np.load(data_test_path+'.npy')

    print("loading test name & slice_id & category......")
    test_info = scio.loadmat(info_test_path + '.mat')

    if use_embedded:
        img_test = test_embedded_data
    else:
        img_test = test_data["img"]

    mask_test = test_data["mask"]
    name_test = test_info["name"]
    slice_id_test = test_info["slice_id"]
    category_test = test_info["category"]

    mydataset_test = Mydataset(mode='test', img=img_test, mask=mask_test, name=name_test, slice_id=slice_id_test,
                                category=category_test, load_from_disk=load_from_disk)


    return mydataset_test

def save_embedded_data():
    
    ##把原始的二维数据使用img encode进行预处理, 节省训练的时间
    from segment_anything.utils.transforms import ResizeLongestSide
    from segment_anything.build_sam import sam_model_registry

    data_root = "../BTCV_dataset1"
    device = "cuda:2"
    train_data_dir = os.path.join(data_root, "pre_processed_dataset1_train.npz")
    val_data_dir = os.path.join(data_root, "pre_processed_dataset1_val.npz")
    test_data_dir = os.path.join(data_root, "pre_processed_dataset1_test.npz")
    train_data = np.load(train_data_dir)
    val_data = np.load(val_data_dir)
    test_data = np.load(test_data_dir)

    train_img = train_data['img']
    val_img = val_data['img']
    test_img = test_data['img']
    
    sam_checkpoint = "../pretrain_model/sam_vit_h.pth"
    sam_model = sam_model_registry['vit_h'](checkpoint=sam_checkpoint).to(device)
    transform = ResizeLongestSide(train_img.shape[-1])

    def embedding_single_img(img):

        img = torch.from_numpy(img).to(device)
        with torch.no_grad():
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            img = img.unsqueeze(1)

            # print(img.shape)
            # pdb.set_trace()
            img = transform.apply_image_torch(img.float())
                    
            ###问题: 输入三通道
            img = img.repeat(1, 3, 1, 1)
            
            input_img = sam_model.preprocess(img) 
            image_embedding = sam_model.image_encoder(input_img)
        image_embedding = np.array(image_embedding.cpu())
        return image_embedding

    ####可以根据自己的gpu显存大小改batch size
    bc = 5

    #training_data

    pbar = tqdm(range(train_img.shape[0]//bc), ncols=90, desc='Train')
    train_data_embedded = []
    for i in pbar:
        train_data_embedded.append(embedding_single_img(train_img[bc*i: bc*(i+1)]))
    train_data_embedded.append(embedding_single_img(train_img[bc * (train_img.shape[0]//bc):]))
    train_data_embedded = np.concatenate(train_data_embedded, axis=0)
    
    train_data_newdir = os.path.join(data_root, "vit-h_embedding_dataset1_train.npy")
    # train_data["img"] = train_data_embedded
    np.save(train_data_newdir, train_data_embedded, allow_pickle=True)

    #val_img

    pbar = tqdm(range(val_img.shape[0]//bc), ncols=90, desc='Val')
    val_data_embedded = []
    for i in pbar:
        val_data_embedded.append(embedding_single_img(val_img[bc*i: bc*(i+1)]))
    if bc * (val_img.shape[0]//bc) < val_img.shape[0]:
        val_data_embedded.append(embedding_single_img(val_img[bc * (val_img.shape[0]//bc):]))
    val_data_embedded = np.concatenate(val_data_embedded, axis=0)
    
    val_data_newdir = os.path.join(data_root, "vit-h_embedding_dataset1_val.npy")
    np.save(val_data_newdir, val_data_embedded, allow_pickle=True)

    #test_img

    pbar = tqdm(range(test_img.shape[0]//bc), ncols=90, desc='Test')
    test_data_embedded = []
    for i in pbar:
        test_data_embedded.append(embedding_single_img(test_img[bc*i: bc*(i+1)]))
    if bc * (test_img.shape[0]//bc) < test_img.shape[0]:
        test_data_embedded.append(embedding_single_img(test_img[bc * (test_img.shape[0]//bc):]))
    test_data_embedded = np.concatenate(test_data_embedded, axis=0)
    
    test_data_newdir = os.path.join(data_root, "vit-h_embedding_dataset1_test.npy")
    np.save(test_data_newdir, test_data_embedded, allow_pickle=True)

if __name__ == '__main__':
    save_embedded_data()
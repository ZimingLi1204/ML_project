import sys

sys.path.append("..")
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import os
import scipy.io as scio
import gc
from tqdm.auto import tqdm
from utils.utils import get_promt
import matplotlib.pyplot as plt

class Maskdataset(Dataset):

    def __init__(self,  img: np.array, mask: np.array, name: list, slice_id: list, category: list):
        '''
        img: N * 512 * 512.
        mask: 每个data的groundtruth 0-1mask, N * 512 * 512
        name: 每个data对应的CT编号 1-10, 21-40   list中每个元素格式为 str(00xx) 如 "0001"      N * 1
        slice_id: 每个data对应的切片编号 每个CT有80-150个切片    list中每个元素格式为 str(xxx)    N * 1
        category: 每个mask对应的类别 范围为1-13     N * 1
        '''
        self.img = img.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.name = name
        self.slice_id = slice_id
        self.category = category.astype(np.int64) - 1
        
        self.img = self.img / 255
        
        # N = self.img.shape[0]
        # self.mean = np.mean(self.img, axis=0)
        # self.std = np.sqrt(np.sum(((self.img - self.mean)**2), axis=0))
        # print(self.mean.shape, self.std.shape)
        # print(self.mean.reshape(1, 512, 512).repeat(N, axis=0).shape)
        # self.img = (self.img-self.mean.reshape(1, 512, 512).repeat(N, axis=0)) / self.std.reshape(1, 512, 512).repeat(N, axis=0)
        
        
        self.data_merge = np.concatenate([self.mask.reshape(-1, 1, 512, 512), self.img.reshape(-1, 1, 512, 512)], axis=1)
        # self.data_merge = np.multiply(self.mask, self.img)
        self.data_merge = self.data_merge.reshape(-1, 2, 512, 512)
        print(self.data_merge.shape)
        # breakpoint()
        
                
        
    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):

        # img = torch.from_numpy(self.img[index]).float()
        # mask = torch.from_numpy(self.mask[index]).float()
        promt = get_promt(self.img[index], self.mask[index], promt_type='box').astype(np.float32)
        data_merge = self.data_merge[index]
        category = self.category[0][index]
        return data_merge, category, promt

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        img_net = []

        # block 1
        #net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, padding=1, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        img_net.append(self.conv1)
        img_net.append(nn.BatchNorm2d(4))
        img_net.append(nn.ReLU())
        img_net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, padding=1, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        img_net.append(self.conv2)
        img_net.append(nn.BatchNorm2d(4))
        img_net.append(nn.ReLU())
        img_net.append(nn.Dropout(p=0.3))
        img_net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(in_features=4*32*32, out_features=16)
        nn.init.kaiming_normal_(self.fc1.weight)
        img_net.append(nn.Flatten())
        img_net.append(self.fc1)
        self.img_net = nn.Sequential(*img_net)
        
        classifier = []
        self.fc2 = nn.Linear(in_features=20, out_features=self.num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)
        classifier.append(self.fc2)
        self.classifier = nn.Sequential(*classifier)
        
        '''
        net.append(nn.Conv2d(in_channels=8, out_channels=8, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=3, stride=3))

        # block 2
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        net.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        net.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        net.append(nn.Flatten())

        net.append(nn.Linear(in_features=3200, out_features=64))
        net.append(nn.ReLU())
        net.append(nn.Dropout(p=0.5))
        net.append(nn.Linear(in_features=64, out_features=self.num_classes))
        net.append(nn.Softmax())
        '''
        # add classifier into class property
        # self.classifier = nn.Sequential(*net)


    def forward(self, data, prompt):
        # data shape = tensro(2, 512, 512)
        # prompt = box(XYXY)    tensor(4, )
        
        feat = self.img_net(data)
        # shape = (32, )
        
        x = torch.cat((feat, prompt), dim=1)
        
        return self.classifier(x)

    def test(self, val_dataloader, bs):

        loss_fn = nn.CrossEntropyLoss()
        acc = 0
        loss = 0
        for data_merge, gt_category, promt in tqdm(val_dataloader, ncols=90, desc="val", position=1):
            with torch.no_grad():
                data_merge = data_merge.to(device)
                gt_category = gt_category.to(device)
                promt = promt.to(device)
                pred_category = self.forward(data_merge, promt)
                loss += loss_fn(pred_category, gt_category)
                pred_category = torch.argmax(pred_category, dim=1)
                for index in range(len(gt_category)):
                    if (pred_category[index] == gt_category[index]):
                        acc += 1

        acc = acc / (len(val_dataloader) * bs)
        loss = loss / len(val_dataloader)


        return acc, loss.detach().cpu()


if __name__ == "__main__":
    cfg_file = open('config/cfg.yaml')
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)
    cfg_file.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    use_checkpoint_asinit = cfg["train"]["use_checkpoint_asinit"]
    
    if use_checkpoint_asinit:
        checkpoint_id = cfg["train"]["checkpoint_id"]
        cnn = torch.load("../Task3/checkpoints/1x512x512_epoch{}.pth".format(checkpoint_id)).to(device)
    else:
        cnn = CNN(num_classes=13).to(device)
    # d1 = torch.randint(0, 1, size=(1, 2, 512, 512)).float().cuda()
    # d2 = torch.tensor([[1, 2, 3, 4]]).cuda()
    # print(cnn(d1, d2).shape)
    
    # summary(cnn, [(2, 512, 512), (4, )])
    

    data_train_path = os.path.join(cfg['data']['data_root'], cfg["data"]["data_name"] + '_' + "train")
    data_val_path = os.path.join(cfg['data']['data_root'], cfg["data"]["data_name"] + '_' + "val")
    info_train_path = os.path.join(cfg['data']['data_root'], cfg["data"]["info_name"] + '_' + "train")
    info_val_path = os.path.join(cfg['data']['data_root'], cfg["data"]["info_name"] + '_' + "val")

    train_data = np.load(info_train_path + '.npz')
    val_data = np.load(info_val_path + '.npz')
    train_info = scio.loadmat(info_train_path + '.mat')
    val_info = scio.loadmat(info_val_path + '.mat')

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
    # del train_data, val_data, train_info, val_info
    # gc.collect()

    dataset_train = Maskdataset(img=img_train, mask=mask_train, name=name_train, slice_id=slice_id_train,
                                category=category_train)
    dataset_val = Maskdataset(img=img_val, mask=mask_val, name=name_val, slice_id=slice_id_val,
                              category=category_val)

    bs = cfg["train"]["batch_size"]
    train_dataloader = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=4)

    print("#######################dataset prepared#########################")



    optimizer = torch.optim.Adam(cnn.parameters(), lr=cfg["train"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=cfg['train']['log_dir'])
    n_iter = 0
    total_acc = []
        
    print("#######################training start#########################")

    cnn.train()
    for epoch in tqdm(range(cfg["train"]["max_epoch"]), ncols=90, desc="epoch", position=0):
        # print(cnn.test(val_dataloader, bs))
        for data_merge, category, promt in tqdm(train_dataloader, ncols=90, desc="train", position=2):
            #print(type(data_merge), type(promt))
            #print(data_merge.dtype, promt.dtype)
            #print(data_merge.shape)
            # print(mask.shape, category.shape)   # (bs, 1, 512, 512)   (bs,)
            # plt.imsave("1000_qwq.jpg", np.full((512, 512), 0))
            # plt.imsave("img_qwq.jpg", img[0])
            # plt.imsave("mask_qwq.jpg", mask[0])
            # plt.imsave("merge_qwq.jpg", data_merge[0][0])
            # breakpoint()
            
            category = category.to(device)
            data_merge = data_merge.to(device)
            promt = promt.to(device)
            
            optimizer.zero_grad()
            output = cnn(data_merge, promt)
            loss = loss_fn(output, category)
            loss.backward()
            optimizer.step()

            n_iter += 1
            writer.add_scalar('loss/train', loss.cpu(), n_iter)
            # if n_iter % 100 == 0:
            #     writer.add_image("raw_data", data_merge[0][0].int(), n_iter, dataformats="HW")
            #     writer.add_image("raw_img", img[0].int(), n_iter, dataformats="HW")
            #     writer.add_image("raw_mask", mask[0].int(), n_iter, dataformats="HW")

        torch.save(cnn, "../Task3/checkpoints/1x512x512_epoch{}.pth".format(epoch+50))
        cnn.eval()
        acc, val_loss = cnn.test(val_dataloader, bs)
        cnn.train()
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", acc, epoch)
        total_acc.append(acc)
        print("test acc = ", acc)

    print("total acc:", total_acc)



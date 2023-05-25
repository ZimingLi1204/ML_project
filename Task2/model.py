import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pdb
sys.path.append("..")
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling import Sam
from torch.nn.functional import threshold, normalize
from torch.utils.tensorboard import SummaryWriter
from segment_anything.utils.transforms import ResizeLongestSide
from utils.pytorch_loss.focal_loss import FocalLossV2
from utils.pytorch_loss.soft_dice_loss import SoftDiceLossV2
from utils.loss import multi_loss

class Mysam(Sam):
    def __init__(self) -> None:
        super.__init__()

class finetune_sam():
    
    def __init__(self, cfg) -> None:
        self.model_type = cfg["model"]['model_type']
        assert self.model_type in ["vit_h", 'vit_l', 'vit_b']
        sam_checkpoint = cfg['model']['model_root'] + '/' + 'sam_' + self.model_type + '.pth'  
        self.device = cfg['device']
        if self.device == 'cuda':
            torch.cuda.set_device(cfg["device_id"])
            self.device += ':' + str(cfg['device_id'])
        self.sam_model = sam_model_registry[self.model_type](checkpoint=sam_checkpoint)
        self.sam_model.to(device=self.device)
       
        self.use_tensorboard = cfg["train"]['use_tensorboard']
        self.log_dir = cfg['train']['log_dir']
        self.use_embedded = cfg["data"]["use_embedded"]

        self.optim = cfg['train']['optimizer']
        self.loss = cfg['train']['loss']
        self.lr = cfg['train']['learning_rate']


        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.sam_model.mask_decoder.parameters(), lr=self.lr, weight_decay=cfg['train']['weight_decay']) 
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.sam_model.mask_decoder.parameters(), lr=self.lr, weight_decay=cfg['train']['weight_decay']) 
        else:
            raise NotImplementedError
        
        if self.loss == 'MSE':    
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss == 'sam_loss':
            self.focal_loss = FocalLossV2()
            self.dice_loss = SoftDiceLossV2()
            self.loss_fn = multi_loss(loss_list = [self.focal_loss, self.dice_loss], weight_list = cfg["train"]["weight_list"])
        else:
            raise NotImplementedError
            
        self.cfg = cfg
        self.transform = ResizeLongestSide(self.cfg['data']['input_size'][0])

        #设定image size
        self.input_size = (self.cfg['data']['input_size'][0], self.cfg['data']['input_size'][1])
        self.original_image_size = (self.cfg['data']['img_size'][0], self.cfg['data']['img_size'][1])

        #load finetuned decoder (if need)
        self.load_model(cfg)

        print("finish initialize model class")

    def load_model(self, cfg):
        if cfg["model"]["load_decoder"]:
            self.sam_model.mask_decoder.load_state_dict(torch.load(cfg["model"]["decoder_path"]))

    def train(self, dataloader, val_dataset, metrics=None):

        ##tensorboard summary writer
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)

        n_iter = 0

        print("############start training################")
        for epoch in tqdm(range(self.cfg['train']['max_epoch']), ncols=90, desc="epoch", position=1):
            
            # ###eval结果并save model
            result = self.val(val_dataset, metrics=metrics)
            dice_val, loss_val, iou_val = result
            if self.use_tensorboard:
                writer.add_scalar('loss/val', loss_val, epoch)
                writer.add_scalar('dice/val', dice_val, epoch)
                writer.add_scalar('iou/val', iou_val, epoch)

            ###save model
            torch.save(self.sam_model.mask_decoder.state_dict(), self.log_dir + '/' + '{}.pth'.format(epoch))
            ###eval end###
            
            pbar = tqdm(dataloader, ncols=90, desc="iter", position=0)
            for img, gt_mask, promt, promt_label, promt_type in pbar:

                #chanve device
                img = img.to(self.device)
                gt_mask = gt_mask.to(self.device).unsqueeze(1).float()
                promt = promt.to(self.device)
                if isinstance(promt_label[0], torch.Tensor):
                    promt_label = promt_label.to(self.device)
                elif promt_label[0] != -1:
                    promt_label = promt_label.to(self.device)
                    
                with torch.no_grad():
                    
                    if self.use_embedded:
                        image_embedding = img
                    else:
                        ####TODO####
                        #把最长边resize成1024, 短边padding
                        img = img.unsqueeze(1) #batch * 1 * input_size * input_size
                        img = self.transform.apply_image_torch(img.float())

                        img = img.repeat(1, 3, 1, 1) #输入三通道
                        input_img = self.sam_model.preprocess(img)
                        image_embedding = self.sam_model.image_encoder(input_img)

                    ###构建promt
                    points, boxes, masks = None, None, None
                    

                    if isinstance(promt_type, tuple):
                        promt_type = promt_type[0]
                    
                    if promt_type == 'box':
                        boxes = promt * self.input_size[0] / self.original_image_size[0]
                    elif promt_type == 'mask':
                        masks = promt
                    elif promt_type == 'points' or promt_type == 'single_point':
                        points = promt * self.input_size[0] / self.original_image_size[0], promt_label
                    else:
                        raise NotImplementedError
                                            
                    
                    #根据promt生成promt embedding
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=points,
                        boxes=boxes,
                        masks=masks,
                    )
                
               #decoder

                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)

                # if promt_type == 'box':
                #     for i in range(upscaled_masks.shape[0]):
                #         coord_mask = torch.ones_like(upscaled_masks, dtype=torch.int8)
                #         coord_mask = 

                ###计算loss, update
                # pdb.set_trace()
                if self.loss == 'MSE':
                    loss = self.loss_fn(binary_mask, gt_mask)
                else:
                    loss = self.loss_fn(upscaled_masks, gt_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                ###log
                pbar.set_postfix(loss = loss.item())
                n_iter += 1
                if self.use_tensorboard:
                    writer.add_scalar('loss/train', loss.cpu(), n_iter)
                    writer.add_scalar('iou/train', iou_predictions.mean().cpu(), n_iter) #因为只设置了一个mask, 所以直接取0


    def val(self, dataset, metrics=None):

        self.sam_model.eval()

        ###调用task1中的val函数
        loss_all = []
        iou_all = []
        mask_all = np.zeros([len(dataset), self.original_image_size[0], self.original_image_size[1]], dtype=np.int8)

        pbar = tqdm(range(len(dataset)), ncols=90, desc="eval", position=0)
        for i in pbar:

            img, gt_mask, promt, promt_label, promt_type = dataset[i]

            ###change format and device
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
            gt_mask = torch.from_numpy(gt_mask).to(self.device).unsqueeze(0).float()
            promt = torch.from_numpy(promt).to(self.device).unsqueeze(0)
            # pdb.set_trace()
            if isinstance(promt_label, torch.Tensor):
                promt_label = torch.from_numpy(promt_label).to(self.device).unsqueeze(0)
            elif promt_label != -1:
                promt_label = torch.from_numpy(promt_label).to(self.device).unsqueeze(0)

            ###use sam model generate mask
            with torch.no_grad():
                
                if self.use_embedded:
                    image_embedding = img
                else:
                    img = img.unsqueeze(1)
                    input_img = self.transform.apply_image_torch(img)#把最长边resize成1024, 短边padding
                    input_img = input_img.repeat(1, 3, 1, 1) #输入三通道
                    input_img = self.sam_model.preprocess(input_img)
                    image_embedding = self.sam_model.image_encoder(input_img)
                
                ###构建promt
                points, boxes, masks = None, None, None
                
                if isinstance(promt_type, list):
                    promt_type = promt_type[0]

                if promt_type == 'box':
                    boxes = promt * self.input_size[0] / self.original_image_size[0]
                elif promt_type == 'mask':
                    masks = promt
                elif promt_type == 'points' or promt_type == 'single_point':
                    points = promt * self.input_size[0] / self.original_image_size[0], promt_label
                else:
                    raise NotImplementedError                    
                
                #根据promt生成promt embedding
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
            
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device).squeeze() #低于0的扔掉, 高于0的除最大值normalize到1

                #loss                
                if self.loss == 'MSE':
                    loss = self.loss_fn(binary_mask, gt_mask)
                else:
                    loss = self.loss_fn(upscaled_masks, gt_mask)
                    
                ###汇总
                loss_all.append(loss.cpu().item())
                iou_all.append(iou_predictions.cpu())
                mask_all[i] = (binary_mask.cpu())


        loss_all = np.array(loss_all).mean() 
        # pdb.set_trace()
        iou_all = torch.cat(iou_all).mean().item()
        mDice = metrics.eval_data_processing(6, mask_all)
        print("val mDice:", np.array(mDice).mean())

        self.sam_model.train()

        return np.array(mDice).mean(), loss_all, iou_all

    def test(self, dataset, metrics=None):

        self.sam_model.eval()

        ###调用task1的test 函数
        loss_all = []
        iou_all = []
        mask_all = np.zeros([len(dataset), self.original_image_size[0], self.original_image_size[1]], dtype=np.int8)

        pbar = tqdm(range(len(dataset)), ncols=90, desc="eval", position=0)
        for i in pbar:

            img, gt_mask, promt, promt_label, promt_type = dataset[i]

            ###change format and device
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
            gt_mask = torch.from_numpy(gt_mask).to(self.device).unsqueeze(0).float()
            promt = torch.from_numpy(promt).to(self.device).unsqueeze(0)
            if promt_label != -1:
                promt_label = torch.from_numpy(promt_label).to(self.device).unsqueeze(0)
            
            ###use sam model generate mask
            with torch.no_grad():
                
                if self.use_embedded:
                    image_embedding = img
                else:
                    img = img.unsqueeze(1)
                    input_img = self.transform.apply_image_torch(img)#把最长边resize成1024, 短边padding
                    input_img = input_img.repeat(1, 3, 1, 1) #输入三通道
                    input_img = self.sam_model.preprocess(input_img)
                    image_embedding = self.sam_model.image_encoder(input_img)
                
                ###构建promt
                points, boxes, masks = None, None, None
                
                if isinstance(promt_type, list):
                    promt_type = promt_type[0]

                if promt_type == 'box':
                    boxes = promt * self.input_size[0] / self.original_image_size[0]
                elif promt_type == 'mask':
                    masks = promt
                elif promt_type == 'points' or promt_type == 'single_point':
                    points = promt * self.input_size[0] / self.original_image_size[0], promt_label
                else:
                    raise NotImplementedError                    
                
                #根据promt生成promt embedding
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
            
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device).squeeze() #低于0的扔掉, 高于0的除最大值normalize到1

                #loss                
                loss = self.loss_fn(binary_mask.squeeze(), gt_mask.squeeze())

                ###汇总
                loss_all.append(loss.cpu().item())
                iou_all.append(iou_predictions.cpu())
                mask_all[i] = (binary_mask.cpu())

        
        loss_all = np.array(loss_all).mean() 
        iou_all = torch.cat(iou_all).mean().item()
        mDice = metrics.eval_data_processing(6, mask_all)
        print("test mDice:", mDice)
        print("test loss:", loss_all)
        print("test iou:", iou_all)
        
        return np.array(mDice).mean(), loss_all, iou_all


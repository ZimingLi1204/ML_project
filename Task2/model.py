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
        else:
            raise NotImplementedError
            
        self.cfg = cfg
        self.transform = ResizeLongestSide(self.cfg['data']['input_size'][0])

        #设定image size
        self.input_size = (self.cfg['data']['input_size'][0], self.cfg['data']['input_size'][1])
        self.original_image_size = (self.cfg['data']['img_size'][0], self.cfg['data']['img_size'][1])

        print("finish initialize model class")

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

                img = img.to(self.device)
                gt_mask = gt_mask.to(self.device).unsqueeze(1).float()
                promt = promt.to(self.device)
                if promt_label != -1:
                    promt_label = promt_label.to(self.device)

                # pdb.set_trace()
                with torch.no_grad():
                    
                    if self.use_embedded:
                        image_embedding = img
                    else:
                        ####TODO####
                        #把最长边resize成1024, 短边padding
                        img = img.unsqueeze(1)
                        # pdb.set_trace()
                        img = self.transform.apply_image_torch(img.float())
                        
                        ###问题: 输入三通道
                        img = img.repeat(1, 3, 1, 1)
                        input_img = self.sam_model.preprocess(img)
                        # pdb.set_trace()
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

                ###计算loss, update
                # pdb.set_trace()
                loss = self.loss_fn(binary_mask, gt_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                ###log
                pbar.set_postfix(loss = loss.item())
                n_iter += 1
                if self.use_tensorboard:
                    writer.add_scalar('loss/train', loss.cpu(), n_iter)
                    writer.add_scalar('iou/train', iou_predictions.mean().cpu(), n_iter) #因为只设置了一个mask, 所以直接取0

        pass

    def val(self, dataset, metrics=None):
        ###调用task1中的val函数
        loss_all = []
        iou_all = []
        mask_all = []

        pbar = tqdm(range(len(dataset)), ncols=90, desc="eval", position=0)
        for i in pbar:

            img, gt_mask, promt, promt_label, promt_type = dataset[i]

            ###change format and device
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
            gt_mask = torch.from_numpy(gt_mask).to(self.device).unsqueeze(0).float()
            promt = torch.from_numpy(promt).to(self.device).unsqueeze(0)
            if promt_label != -1:
                promt_label = torch.from_numpy(promt_label).to(self.device).unsqueeze(0)

            with torch.no_grad():
                
                if self.use_embedded:
                    image_embedding = img
                else:
                    ####TODO####
                    #把最长边resize成1024, 短边padding
                    img = img.unsqueeze(1)
                    # pdb.set_trace()
                    img = self.transform.apply_image_torch(img.float())
                    
                    ###问题: 输入三通道
                    img = img.repeat(1, 3, 1, 1)
                    input_img = self.sam_model.preprocess(img)
                    # pdb.set_trace()
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
            
                # pdb.set_trace()
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device).squeeze()
                
                mask_all.append(binary_mask)

                loss = self.loss_fn(binary_mask, gt_mask.squeeze())

                loss_all.append(loss.cpu().item())
                iou_all.append(iou_predictions.cpu())

        ###save model
        # torch.save()
        # pdb.set_trace()
        mask_all = np.array(torch.stack(mask_all, dim=0).cpu())
        loss_all = np.array(loss_all).mean() 
        iou_all = torch.concatenate(iou_all).mean().item()

        mDice = metrics.eval_data_processing(6, mask_all)
        print("val mDice:", mDice)

        return np.array(mDice).mean(), loss_all, iou_all

    def test(self, dataset, metrics=None):
        ###调用task1的test 函数
        loss_all = []
        iou_all = []
        mask_all = np.zeros([len(dataset), self.original_image_size[0], self.original_image_size[1]])

        pbar = tqdm(range(len(dataset)), ncols=90, desc="eval", position=0)
        for i in pbar:

            img, gt_mask, promt, promt_label, promt_type = dataset[i]

            ###change format and device
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
            gt_mask = torch.from_numpy(gt_mask).to(self.device).unsqueeze(0).float()
            promt = torch.from_numpy(promt).to(self.device).unsqueeze(0)
            if promt_label != -1:
                promt_label = torch.from_numpy(promt_label).to(self.device).unsqueeze(0)
            
            # pdb.set_trace()
            with torch.no_grad():
                
                if self.use_embedded:
                    image_embedding = img
                else:
                    ####TODO####
                    #把最长边resize成1024, 短边padding
                    img = img.unsqueeze(1)
                    # pdb.set_trace()
                    input_img = self.transform.apply_image_torch(img)
                    
                    ###问题: 输入三通道
                    input_img = input_img.repeat(1, 3, 1, 1)
                    input_img = self.sam_model.preprocess(input_img)
                    # pdb.set_trace()
                    image_embedding = self.sam_model.image_encoder(input_img)
                
                # pdb.set_trace()
            
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
                # pdb.set_trace()
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
            
                # pdb.set_trace()
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
               
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device).squeeze()
                
                mask_all[i] = (binary_mask.cpu())

                loss = self.loss_fn(binary_mask.squeeze(), gt_mask.squeeze())

                loss_all.append(loss.cpu().item())
                iou_all.append(iou_predictions.cpu())
            # pdb.set_trace()

        ###save model
        # torch.save()
        
        
        loss_all = np.array(loss_all).mean() 
        iou_all = torch.concatenate(iou_all).mean().item()

        # pdb.set_trace()
        mDice = metrics.eval_data_processing(6, mask_all)
        print("test mDice:", mDice)
        print("test loss:", loss_all)
        print("test iou:", iou_all)
        
        return np.array(mDice).mean(), loss_all, iou_all


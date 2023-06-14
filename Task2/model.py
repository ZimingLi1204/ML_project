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
from utils.metrics import dice_coefficient
from torch.optim.lr_scheduler import LinearLR, StepLR


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
        self.linear_warmup = cfg["train"]["linear_warmup"]
        self.lr_decay = cfg["train"]["lr_decay"]
        self.multimask = cfg['train']['multimask']

        ###optimizer
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.sam_model.mask_decoder.parameters(), lr=self.lr, weight_decay=cfg['train']['weight_decay']) 
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.sam_model.mask_decoder.parameters(), lr=self.lr, weight_decay=cfg['train']['weight_decay']) 
        else:
            raise NotImplementedError

        ###loss
        if self.loss == 'MSE':    
            self.loss_fn = nn.MSELoss()
        elif self.loss == 'sam_loss':
            self.focal_loss = FocalLossV2()
            self.dice_loss = SoftDiceLossV2()
            self.loss_fn = multi_loss(loss_list = [self.focal_loss, self.dice_loss], weight_list = cfg["train"]["weight_list"], device=self.device)
        else:
            raise NotImplementedError
        self.iou_loss_fn = nn.MSELoss()
            
        ###Training tricks
        if self.linear_warmup:
            self.warmup_scheduler = LinearLR(self.optimizer, start_factor=cfg["train"]["start_factor"], total_iters=cfg["train"]["warmup_iter"])

        if self.lr_decay:
            if cfg["train"]["lr_schedular"] == "StepLR":
                self.decay_schedular = StepLR(self.optimizer, step_size=cfg["train"]["step_size"], gamma=cfg["train"]["schedular_gamma"])
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
            dice_val, loss_val, mask_loss_val, iou_loss_val, iou_val = result
            if self.use_tensorboard:
                writer.add_scalar('loss/val', loss_val, epoch)
                writer.add_scalar('loss/mask_loss_val', mask_loss_val, epoch)
                writer.add_scalar('loss/iou_val', iou_loss_val, epoch)
                writer.add_scalar('dice/val', dice_val, epoch)
                writer.add_scalar('iou/val', iou_val, epoch)

            ###save model
            if epoch > 0:
                torch.save(self.sam_model.mask_decoder.state_dict(), self.log_dir + '/' + '{}.pth'.format(epoch))

            ###eval end###
            
            pbar = tqdm(dataloader, ncols=90, desc="iter", position=0)
            for img, gt_mask, promt, promt_label, promt_type in pbar:

                #chanve device
                img = img.to(self.device)
                gt_mask = gt_mask.to(self.device).unsqueeze(1).float()
                promt = promt.to(self.device)

                if torch.is_tensor(promt_label):
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
                        boxes = promt * (self.input_size[0] / self.original_image_size[0])
                    elif promt_type == 'mask':
                        masks = promt
                    elif promt_type == 'points' or promt_type == 'single_point' or promt_type == "grid_points":
                        points = promt * (self.input_size[0] / self.original_image_size[0]), promt_label
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
                    multimask_output=self.multimask,
                )

                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)

                ###计算loss, update
                # pdb.set_trace()
                # iou = torch.sum(binary_mask * gt_mask.unsqueeze(1), dim=(-1, -2))
                # _, max_idx = torch.max(iou, dim=1)
                # binary_mask = 
                # pdb.set_trace()
                # if self.multimask is not None :
                #     mask_loss = torch.zeros(upscaled_masks.shape[0], device=self.device)
                #     if self.loss == 'MSE':
                        
                #         for i in range(mask_loss.shape[0]):
                #             mask_loss0 = self.loss_fn(binary_mask[i, 0].unsqueeze(0), gt_mask[i])
                #             mask_loss1 = self.loss_fn(binary_mask[i, 1].unsqueeze(0), gt_mask[i])
                #             mask_loss2 = self.loss_fn(binary_mask[i, 2].unsqueeze(0), gt_mask[i])
                #             loss_i = torch.concat([mask_loss0, mask_loss1, mask_loss2])
                #             if self.multimask == 'min':
                #                 mask_loss[i], _ = torch.min(loss_i, dim=1)
                #             elif self.multimask == 'max':
                #                 mask_loss, _ = torch.max(loss_i, dim=1)
                #             elif self.multimask == 'mean':
                #                 mask_loss, _ = torch.mean(loss_i, dim=1)
                #             else: 
                #                 raise NotImplementedError
                #     else:
                #         # mask_loss = torch.zeros((upscaled_masks.shape[0]), device=self.device)
                #         for i in range(mask_loss.shape[0]):
                #             mask_loss0 = self.loss_fn(upscaled_masks[i, 0].unsqueeze(0), gt_mask[i])
                #             mask_loss1 = self.loss_fn(upscaled_masks[i, 1].unsqueeze(0), gt_mask[i])
                #             mask_loss2 = self.loss_fn(upscaled_masks[i, 2].unsqueeze(0), gt_mask[i])

                #             loss_i = torch.concat([mask_loss0, mask_loss1, mask_loss2])

                #             if self.multimask == 'min':
                #                 mask_loss[i], _ = torch.min(loss_i, dim=1)
                #             elif self.multimask == 'max':
                #                 mask_loss, _ = torch.max(loss_i, dim=1)
                #             elif self.multimask == 'mean':
                #                 mask_loss, _ = torch.mean(loss_i, dim=1)
                #             else: 
                #                 raise NotImplementedError


                #     # pdb.set_trace()

                #     mask_loss = mask_loss.mean()
                
                if self.loss == 'MSE':
                    mask_loss = self.loss_fn(binary_mask, gt_mask, multimask=self.multimask)
                else:
                    mask_loss = self.loss_fn(upscaled_masks, gt_mask, multimask=self.multimask)

                #iou and iou loss 
                # pdb.set_trace()
                iou = torch.sum(binary_mask * gt_mask.repeat(1, binary_mask.shape[1], 1, 1), dim=[-1, -2]) / torch.sum(gt_mask.repeat(1, binary_mask.shape[1], 1, 1), dim=[-1, -2])
                
                iou_loss = self.iou_loss_fn(iou_predictions, iou) 
                
                loss = mask_loss + iou_loss * self.cfg["train"]["iou_scale"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.linear_warmup and n_iter < self.cfg["train"]["warmup_iter"]:
                    self.warmup_scheduler.step()
                elif self.lr_decay:
                    self.decay_schedular.step()

                #dice_coef
                dice_coef = (2 * torch.sum((binary_mask * gt_mask), dim=[-1, -2]) / (torch.sum(binary_mask, dim=[-2, -1]) + torch.sum(gt_mask, dim=[-2, -1]))).mean() 
                
                # pdb.set_trace()

                ###log
                pbar.set_postfix(loss = loss.item())
                n_iter += 1
                if self.use_tensorboard:
                    writer.add_scalar('loss/train', loss.cpu(), n_iter)
                    writer.add_scalar('loss/train_mask_loss', mask_loss.cpu(), n_iter)
                    writer.add_scalar('loss/train_iou_loss', iou_loss.cpu(), n_iter)
                    writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'] , n_iter)
                    writer.add_scalar('iou/train', iou.mean().cpu(), n_iter) 
                    writer.add_scalar('dice/train', dice_coef.cpu(), n_iter) #因为只设置了一个mask, 所以直接取0


    def val(self, dataset, metrics=None):

        self.sam_model.eval()

        ###调用task1中的val函数
        loss_all = []
        iou_loss_all = []
        mask_loss_all = []
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
            if torch.is_tensor(promt_label):
                promt_label = promt_label.to(self.device).unsqueeze(0)
            elif type(promt_label) is np.ndarray:
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
                
                if isinstance(promt_type, tuple):
                    promt_type = promt_type[0]

                if promt_type == 'box':
                    boxes = promt * (self.input_size[0] / self.original_image_size[0])
                elif promt_type == 'mask':
                    masks = promt
                elif promt_type == 'points' or promt_type == 'single_point' or promt_type == "grid_points":
                    points = promt * (self.input_size[0] / self.original_image_size[0]), promt_label
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
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device).squeeze() #低于0的扔掉, 高于0normalize到1

                #loss
                if self.loss == 'MSE':
                    mask_loss = self.loss_fn(binary_mask, gt_mask)
                else:
                    mask_loss = self.loss_fn(upscaled_masks, gt_mask)
                
                #iou and iou loss 
                iou = torch.sum(binary_mask * gt_mask, dim=[-1, -2]) / torch.sum(gt_mask, dim=[-1, -2])
                iou_loss = self.iou_loss_fn(iou_predictions, iou) 
                
                loss = mask_loss + iou_loss * self.cfg["train"]["iou_scale"]
                
                ###汇总
                loss_all.append(loss.cpu().item())
                mask_loss_all.append(mask_loss.cpu().item())
                iou_loss_all.append(iou_loss.cpu().item())
                iou_all.append(iou.mean().cpu().item())
                mask_all[i] = (binary_mask.cpu())


        loss_all = np.array(loss_all).mean() 
        iou_loss_all = np.array(iou_loss_all).mean()
        mask_loss_all = np.array(mask_loss_all).mean()  
        # pdb.set_trace()
        iou_all = np.array(iou_all).mean()
        mDice = metrics.eval_data_processing(6, mask_all)
        print("val mDice:", np.array(mDice).mean())
        print("val loss:", loss_all)
        print("val iou:", iou_all)
        self.sam_model.train()

        return np.array(mDice).mean(), loss_all, mask_loss_all, iou_loss_all, iou_all

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
            
            if torch.is_tensor(promt_label):
                promt_label = promt_label.to(self.device).unsqueeze(0)
            elif type(promt_label) is np.ndarray:
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
                # pdb.set_trace()
                if isinstance(promt_type, tuple):
                    promt_type = promt_type[0]

                if promt_type == 'box':
                    boxes = promt * (self.input_size[0] / self.original_image_size[0])
                elif promt_type == 'mask':
                    masks = promt
                elif promt_type == 'points' or promt_type == 'single_point' or promt_type == "grid_points":
                    points = promt * (self.input_size[0] / self.original_image_size[0]), promt_label
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
        print("test mDice mean:", np.mean(np.array(mDice)))
        print("test loss:", loss_all)
        print("test iou:", iou_all)
        
        return np.array(mDice).mean(), loss_all, iou_all


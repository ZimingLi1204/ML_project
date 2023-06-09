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
from torch.optim.lr_scheduler import LinearLR
from Task3.testt import Maskdataset, CNN

class Mysam(Sam):
    def __init__(self) -> None:
        super.__init__()

class sam_classifier():
    
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
        
        '''
            添加 category query
        '''
        self.classifier_type = cfg["model"]["classifier_type"]
        if self.classifier_type == "cnn":
            self.classifier = CNN(num_classes=13)
        self.classifier.to(self.device)
        
        self.use_tensorboard = cfg["train"]['use_tensorboard']
        self.log_dir = cfg['train']['log_dir'] # 存 tensorboard 结果
        self.use_embedded = cfg["data"]["use_embedded"]
        self.checkpoints_dir = cfg["train"]["checkpoints_dir"]

        self.optim = cfg['train']['optimizer']
        self.loss = cfg['train']['loss']
        self.lr = cfg['train']['learning_rate']
        self.linear_warmup = cfg["train"]["linear_warmup"]

        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam([{"params": self.sam_model.mask_decoder.parameters()},
                                               {"params": self.classifier.parameters()}], lr=self.lr, weight_decay=cfg['train']['weight_decay']) 
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW([{"params": self.sam_model.mask_decoder.parameters()},
                                               {"params": self.classifier.parameters()}], lr=self.lr, weight_decay=cfg['train']['weight_decay']) 
        else:
            raise NotImplementedError
        
        if self.linear_warmup:
            self.scheduler = LinearLR(self.optimizer, start_factor=cfg["train"]["start_factor"], total_iters=cfg["train"]["warmup_iter"])

        if self.loss == 'MSE':    
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss == 'sam_loss':
            self.focal_loss = FocalLossV2()
            self.dice_loss = SoftDiceLossV2()
            self.loss_fn = multi_loss(loss_list = [self.focal_loss, self.dice_loss], weight_list = cfg["train"]["weight_list"])
        else:
            raise NotImplementedError
        # 使用 cross_entropy lambda 为其权重
        self.classifier_loss_fn = nn.CrossEntropyLoss()
        self.lambda_classifier = cfg["train"]["lambda_classifier"]
            
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
        if cfg["model"]["load_classifier"]:
            classifier_path = cfg["model"]["classifier_path"]
            self.classifier.load_state_dict(torch.load(classifier_path), strict = False)
            
        
    def train(self, dataloader, val_dataloader, metrics=None):

        ##tensorboard summary writer
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)

        n_iter = 0

        print("############start training################")
        for epoch in tqdm(range(self.cfg['train']['max_epoch']), ncols=90, desc="epoch", position=1):
            
            # ###eval结果并save model
            result = self.val(val_dataloader, metrics=metrics)
            dice_val, loss_val, iou_val, acc = result
            if self.use_tensorboard:
                writer.add_scalar('loss/val', loss_val, epoch)
                writer.add_scalar('dice/val', dice_val, epoch)
                writer.add_scalar('iou/val', iou_val, epoch)
                writer.add_scalar('acc/val', acc, epoch)

            ###save model
            if epoch > 0:
                # torch.save(self.sam_model.mask_decoder.state_dict(), self.checkpoints_dir + '/' + 'mask_decoder_epoch{}.pth'.format(epoch))
                torch.save(self.classifier.state_dict(), self.checkpoints_dir + '/' + '{}_epoch_{}.pth'.format(self.classifier_type, epoch))
                
            ###eval end###
            
            pbar = tqdm(dataloader, ncols=90, desc="iter", position=0)
            for img, image_embedding, gt_mask, promt, promt_label, promt_type, category, data_merge in pbar:

                #chanve device
                gt_mask = gt_mask.to(self.device).unsqueeze(1).float()
                promt = promt.to(self.device)
                category = category.long().squeeze().to(self.device)
                # data_merge = data_merge.to(self.device)
                img = img.to(self.device)
                
                if torch.is_tensor(promt_label):
                    promt_label = promt_label.to(self.device)
                elif promt_label[0] != -1:
                    promt_label = promt_label.to(self.device)
                    
                with torch.no_grad():
                    
                    if self.use_embedded:
                        image_embedding = image_embedding.to(self.device)
                    else:
                        ####TODO####
                        #把最长边resize成1024, 短边padding
                        # img = img.to(self.device)
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
                    multimask_output=False,
                )

                #mask
                upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, self.input_size, self.original_image_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)

                # print("qwq", binary_mask.shape, "qwq") # (bs, 1, 512, 512)
                data_merge = torch.cat([binary_mask.detach().reshape(-1, 1, 512, 512), img.reshape(-1, 1, 512, 512)], dim=1)
                data_merge = data_merge.reshape(-1, 2, 512, 512)
                # print(data_merge.shape, "qaq\n") # (bs, 2, 512, 512)

                # classification
                category_result = self.classifier(data_merge, promt)
                # print("\ncategory.shape\n", category.shape)
                classifier_loss = self.lambda_classifier*self.classifier_loss_fn(category_result, category)
                
                ###计算loss, update
                # pdb.set_trace()
                if self.loss == 'MSE':
                    loss = self.loss_fn(binary_mask, gt_mask)
                else:
                    loss = self.loss_fn(upscaled_masks, gt_mask)
                loss = loss + classifier_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.linear_warmup and n_iter < self.cfg["train"]["warmup_iter"]:
                    self.scheduler.step()

                #iou 
                iou = (binary_mask * gt_mask).sum() / gt_mask.sum()

                #dice_coef
                dice_coef = (2 * torch.sum((binary_mask * gt_mask), dim=[-1, -2]) / (torch.sum(binary_mask, dim=[-2, -1]) + torch.sum(gt_mask, dim=[-2, -1]))).mean() 
                
                # pdb.set_trace()

                ###log
                pbar.set_postfix(loss = loss.item())
                n_iter += 1
                if self.use_tensorboard:
                    writer.add_scalar('loss/train', loss.cpu(), n_iter)
                    writer.add_scalar('loss/train_classifier', classifier_loss.cpu(), n_iter)
                    writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'] , n_iter)
                    writer.add_scalar('iou/train', iou.cpu(), n_iter) #因为只设置了一个mask, 所以直接取0
                    writer.add_scalar('dice/train', dice_coef.cpu(), n_iter) #因为只设置了一个mask, 所以直接取0


    def val(self, dataset, metrics=None):

        self.sam_model.eval()
        self.classifier.eval()

        ###调用task1中的val函数
        acc = 0
        loss_all = []
        iou_all = []
        mask_all = np.zeros([len(dataset), self.original_image_size[0], self.original_image_size[1]], dtype=np.int8)

        pbar = tqdm(range(len(dataset)), ncols=90, desc="eval", position=0)
        for i in pbar:
            
            img, image_embedding, gt_mask, promt, promt_label, promt_type, category, data_merge = dataset[i]
            
            ###change format and device
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
            gt_mask = torch.from_numpy(gt_mask).to(self.device).unsqueeze(0).float()
            promt = torch.from_numpy(promt).to(self.device).unsqueeze(0)
            # data_merge = torch.from_numpy(data_merge).to(self.device).unsqueeze(0)
            category = torch.Tensor(category).long().to(self.device) #.unsqueeze(0)
            
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
                    image_embedding = torch.from_numpy(image_embedding).to(self.device).unsqueeze(0)
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

                data_merge = torch.cat([binary_mask.detach().reshape(-1, 1, 512, 512), img.reshape(-1, 1, 512, 512)], dim=1)
                data_merge = data_merge.reshape(-1, 2, 512, 512)
                
                
                # classification
                category_result = self.classifier(data_merge, promt)
                
                pred_category = torch.argmax(category_result, dim=1)
                # print(pred_category.shape, pred_category[0], category[0], "pre_catye\n")
                if (pred_category[0] == category[0]):
                    acc += 1
                
                #loss                
                if self.loss == 'MSE':
                    loss = self.loss_fn(binary_mask, gt_mask)
                else:
                    loss = self.loss_fn(upscaled_masks, gt_mask)
                # print(category_result.shape, category.shape)
                # print(category_result)
                # print(category)
                loss = loss + self.lambda_classifier*self.classifier_loss_fn(category_result, category)
                    
                ###iou
                iou = (binary_mask * gt_mask).sum() / gt_mask.sum()
                
                ###汇总
                loss_all.append(loss.cpu().item())
                iou_all.append(iou.cpu().item())
                mask_all[i] = (binary_mask.cpu())


        loss_all = np.array(loss_all).mean() 
        # pdb.set_trace()
        iou_all = np.array(iou_all).mean()
        mDice = metrics.eval_data_processing(6, mask_all)
        acc = acc / len(dataset)
        
        print("val mDice:", np.array(mDice).mean())
        print("val acc", acc)
        
        self.sam_model.train()
        self.classifier.train()

        return np.array(mDice).mean(), loss_all, iou_all, acc



    def test(self, dataset, metrics=None):
        
        # load model weights
        self.sam_model.mask_decoder.load_state_dict(torch.load(self.cfg["model"]["decoder_path"]))
        classifier_path = self.cfg["model"]["classifier_path"]
        self.classifier.load_state_dict(torch.load(classifier_path), strict = True)
            
        self.sam_model.eval()
        self.classifier.eval()

        ###调用task1中的val函数
        acc = 0
        loss_all = []
        iou_all = []
        mask_all = np.zeros([len(dataset), self.original_image_size[0], self.original_image_size[1]], dtype=np.int8)

        pbar = tqdm(range(len(dataset)), ncols=90, desc="eval", position=0)
        for i in pbar:
            
            img, image_embedding, gt_mask, promt, promt_label, promt_type, category, data_merge = dataset[i]
            
            ###change format and device
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
            gt_mask = torch.from_numpy(gt_mask).to(self.device).unsqueeze(0).float()
            promt = torch.from_numpy(promt).to(self.device).unsqueeze(0)
            # data_merge = torch.from_numpy(data_merge).to(self.device).unsqueeze(0)
            category = torch.Tensor(category).long().to(self.device) #.unsqueeze(0)
            
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
                    image_embedding = torch.from_numpy(image_embedding).to(self.device).unsqueeze(0)
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

                data_merge = torch.cat([binary_mask.detach().reshape(-1, 1, 512, 512), img.reshape(-1, 1, 512, 512)], dim=1)
                data_merge = data_merge.reshape(-1, 2, 512, 512)
                
                
                # classification
                category_result = self.classifier(data_merge, promt)
                
                pred_category = torch.argmax(category_result, dim=1)
                # print(pred_category.shape, pred_category[0], category[0], "pre_catye\n")
                if (pred_category[0] == category[0]):
                    acc += 1
                
                #loss                
                if self.loss == 'MSE':
                    loss = self.loss_fn(binary_mask, gt_mask)
                else:
                    loss = self.loss_fn(upscaled_masks, gt_mask)
                # print(category_result.shape, category.shape)
                # print(category_result)
                # print(category)
                loss = loss + self.lambda_classifier*self.classifier_loss_fn(category_result, category)
                    
                ###iou
                iou = (binary_mask * gt_mask).sum() / gt_mask.sum()
                
                ###汇总
                loss_all.append(loss.cpu().item())
                iou_all.append(iou.cpu().item())
                mask_all[i] = (binary_mask.cpu())


        loss_all = np.array(loss_all).mean() 
        # pdb.set_trace()
        iou_all = np.array(iou_all).mean()
        mDice = metrics.eval_data_processing(6, mask_all)
        acc = acc / len(dataset)
        
        print("test mDice:", np.array(mDice).mean())
        print("test acc", acc)
        
        return np.array(mDice).mean(), loss_all, iou_all, acc

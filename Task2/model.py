import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb
sys.path.append("..")
from segment_anything.build_sam import sam_model_registry
from torch.nn.functional import threshold, normalize
from torch.utils.tensorboard import SummaryWriter
from segment_anything.utils.transforms import ResizeLongestSide

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
       
        self.optim = cfg['train']['optimizer']
        self.loss = cfg['train']['loss']
        self.use_tensorboard = cfg["train"]['use_tensorboard']
        self.log_dir = cfg['train']['log_dir']
        self.lr = cfg['train']['learning_rate']

        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.sam_model.mask_decoder.parameters(), lr=self.lr) 
        else:
            raise NotImplementedError
        if self.loss == 'MSE':    
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError
            
        self.cfg = cfg
        self.transform = ResizeLongestSide(self.cfg['data']['input_size'][0])

        print("finish initialize model class")

    def train(self, dataloader, val_dataloader):
        
        #设定image size
        self.input_size = (self.cfg['data']['input_size'][0], self.cfg['data']['input_size'][1])
        self.original_image_size = (self.cfg['data']['img_size'][0], self.cfg['data']['img_size'][1])

        ##tensorboard summary writer
        
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)

        n_iter = 0

        print("############start training################")
        for epoch in tqdm(range(self.cfg['train']['max_epoch']), ncols=80, desc="epoch", position=0):
            pbar = tqdm(dataloader, ncols=80, desc="iter", position=1)
            for img, gt_mask, promt, promt_label, promt_type in pbar:

                img = img.to(self.device)
                gt_mask = gt_mask.to(self.device).unsqueeze(1).float()
                promt = promt.to(self.device)
                promt_label = promt_label.to(self.device)

                
                with torch.no_grad():
                    
                    ####TODO####
                    #把最长边resize成1024, 短边padding
                    img = img.unsqueeze(1)
                    img = self.transform.apply_image_torch(img).float()
                    # pdb.set_trace()
                    ###问题: 输入三通道
                    img = img.repeat(1, 3, 1, 1)
                    input_img = self.sam_model.preprocess(img)
                    # pdb.set_trace()
                    image_embedding = self.sam_model.image_encoder(input_img)

                    ###构建promt
                    points, boxes, masks = None, None, None
                    
                    if promt_type[0] == 'box':
                        boxes = promt
                    elif promt_type[0] == 'mask':
                        masks = promt
                    elif promt_type[0] == 'points':
                        points = promt, promt_label
                    elif promt_type[0] == 'single_point':
                        points = promt, promt_label
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
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)

                ###计算loss, update
                # pdb.set_trace()
                loss = self.loss_fn(binary_mask, gt_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                ###log
                pbar.set_postfix(loss = loss)
                n_iter += 1
                if self.use_tensorboard:
                    writer.add_scalar('loss/train', loss.cpu(), n_iter)
                    writer.add_scalar('iou/train', iou_predictions.mean().cpu(), n_iter) #因为只设置了一个mask, 所以直接取0


            ###每个epoch后eval结果并save model
            result = self.val(val_dataloader)
            dice_val, loss_val = result
            if self.use_tensorboard:
                writer.add_scalar('loss/val', loss_val.cpu(), epoch)
                writer.add_scalar('dice/val', dice_val.cpu(), epoch)
            
            ###save model
            torch.save(self.sam_model.mask_decoder.parameters(), self.log_dir + '/' + '{}.pth'.format(epoch))

        pass

    def val(self, dataloader):
        ###调用task1中的val函数
        

        ###save model
        torch.save()
        pass

    def test(self, dataloader):
        ###调用task1的test 函数
        pass

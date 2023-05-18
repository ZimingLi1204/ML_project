import sys
import torch
import torch.nn as nn
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from torch.nn.functional import threshold, normalize
from torch.utils.tensorboard import SummaryWriter
from segment_anything.utils.transforms import ResizeLongestSide

class finetune_sam():
    
    def __init__(self, cfg) -> None:
        self.mode_type = cfg["model"]['model_type']
        assert self.model_type in ["vit_h", 'vit_l', 'vit_b']
        sam_checkpoint = 'sam_' + self.model_type + '.pth'  
        self.device = cfg['device']
        self.sam_model = sam_model_registry[self.model_type](checkpoint=sam_checkpoint)
        self.sam_model.to(device=self.device)
       
        self.optim = cfg['train']['optimizer']
        self.loss = cfg['train']['loss']
        self.use_tensorboard = cfg["train"]['use_tensorboard']
        self.log_dir = cfg['train']['log_dir']

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

    def train(self, dataloader, val_dataloader):
        
        #设定image size
        self.input_size = (self.cfg['data']['input_size'][0], self.cfg['data']['input_size'][1])
        self.original_image_size = (self.cfg['data']['img_size'][0], self.cfg['data']['img_size'][1])

        ##tensorboard summary writer
        
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)

        n_iter = 0

        for epoch in range(self.cfg['train']['max_epoch']):
            for img, gt_mask, promt, promt_type in dataloader:
                with torch.no_grad():

                    ####TODO####
                    #把最长边resize成1024, 短边padding
                    img = self.transform(img)
                    image_embedding = self.sam_model.image_encoder(img)

                    ###构建promt
                    points, boxes, masks = None, None, None
                    if promt_type == 'box':
                        boxes = promt
                    elif promt_type == 'mask':
                        masks = promt
                    elif promt_type == 'points':
                        points =promt
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
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)

                ###计算loss, update
                loss = self.loss_fn(binary_mask, gt_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ###log
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

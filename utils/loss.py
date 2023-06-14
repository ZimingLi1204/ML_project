import torch.nn as nn
import pdb
import torch

class multi_loss():
    def __init__(self, loss_list, weight_list, device='cpu'):
        assert len(loss_list) == len(weight_list)
        self.loss_list = loss_list
        self.weight_list = weight_list
        weight_sum = sum(weight_list)
        self.weight_list = [w / weight_sum for w in self.weight_list]
        self.device=device

    def __call__(self, logits, labels, multimask=None):

        if multimask is None:
            return self.calu_loss(logits, labels)    
            
        else:
            mask_loss = torch.zeros(logits.shape[0], device=self.device)
                
            for i in range(mask_loss.shape[0]):
                mask_loss0 = self.calu_loss(logits[i, 0].unsqueeze(0), labels[i]).reshape(1)
                mask_loss1 = self.calu_loss(logits[i, 1].unsqueeze(0), labels[i]).reshape(1)
                mask_loss2 = self.calu_loss(logits[i, 2].unsqueeze(0), labels[i]).reshape(1)
                # pdb.set_trace()
                loss_i = torch.concat([mask_loss0, mask_loss1, mask_loss2])
                if multimask == 'min':
                    mask_loss[i]= torch.min(loss_i)
                elif multimask == 'max':
                    mask_loss[i] = torch.max(loss_i)
                elif multimask == 'mean':
                    mask_loss[i] = torch.mean(loss_i)
                else: 
                    raise NotImplementedError
                # pdb.set_trace()

            mask_loss = mask_loss.mean()
            return mask_loss

    def calu_loss(self, logits, labels):

        for i, loss_fn in enumerate(self.loss_list):
            if (i == 0):
                total_loss = loss_fn(logits, labels) * self.weight_list[i]
                # pdb.set_trace()
            else:
                total_loss += loss_fn(logits, labels) * self.weight_list[i]
                
        return total_loss
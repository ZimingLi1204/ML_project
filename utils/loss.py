import torch.nn as nn
import pdb

class multi_loss():
    def __init__(self, loss_list, weight_list):
        assert len(loss_list) == len(weight_list)
        self.loss_list = loss_list
        self.weight_list = weight_list
      

    def __call__(self, logits, labels):

        for i, loss_fn in enumerate(self.loss_list):
            if (i == 0):
                total_loss = loss_fn(logits, labels) * self.weight_list[i]
                # pdb.set_trace()
            else:
                
                total_loss += loss_fn(logits, labels) * self.weight_list[i]
        
        return total_loss

import torch.nn as nn
import pdb
import torch

class multi_loss():
    def __init__(self, loss_list, weight_list):
        assert len(loss_list) == len(weight_list)
        self.loss_list = loss_list
        self.weight_list = weight_list
        weight_sum = sum(weight_list)
        self.weight_list = [w / weight_sum for w in self.weight_list]

    def __call__(self, logits, labels, raw=False):

        for i, loss_fn in enumerate(self.loss_list):
            if (i == 0):
                total_loss = loss_fn(logits, labels) * self.weight_list[i]
                # pdb.set_trace()
            else:
                
                total_loss += loss_fn(logits, labels) * self.weight_list[i]
        
        return total_loss

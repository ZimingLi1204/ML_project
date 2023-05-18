import torch
import numpy as np
import os
import random

def set_seed(seed, torch_deterministic = False):
    if seed == -1 and torch_deterministic:
        seed = 2333
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed) # random seed for random module
    np.random.seed(seed) # for np module
    torch.manual_seed(seed) # for pytorch module
    os.environ['PYTHONHASHSEED'] = str(seed) # for os env Python hash seed
    torch.cuda.manual_seed(seed) # cuda manual seed
    torch.cuda.manual_seed_all(seed) # cuda manual seed all

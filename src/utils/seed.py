import torch
import random
import numpy as np

#seed definition            
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    #To make CUDA more deterministic (lower performance)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
#%%

import torch, random
import numpy as np
from time import sleep 
from math import floor

from utils import args
from agent import Agent



np.random.seed(args.init_seed)
random.seed(args.init_seed)
torch.manual_seed(args.init_seed)
torch.cuda.manual_seed(args.init_seed)

agent = Agent(GUI = True, args = args)
agent.training()
# %%

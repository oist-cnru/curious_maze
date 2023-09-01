#%%

import torch, random
import numpy as np

from utils import args
from agent import Agent



# Set random seed.
np.random.seed(args.init_seed)
random.seed(args.init_seed)
torch.manual_seed(args.init_seed)
torch.cuda.manual_seed(args.init_seed)

# Make agent, train agent. 
agent = Agent(GUI = True, args = args)
agent.training()
# %%

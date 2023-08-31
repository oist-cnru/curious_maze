#%% 

import argparse, ast
from math import exp, pi

import torch
from torch import nn 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def literal(arg_string): return(ast.literal_eval(arg_string))

# Meta 
parser.add_argument("--init_seed",          type=float,      default = 777)
parser.add_argument('--device',             type=str,        default = device)

# Maze 
parser.add_argument('--arena_list',          type=literal,   default = ["t"])
parser.add_argument('--max_steps',          type=int,        default = 25)
parser.add_argument('--step_lim_punishment',type=float,      default = -1)
parser.add_argument('--wall_punishment',    type=float,      default = -1)
parser.add_argument('--default_reward',     type=literal,    default = [(1, 1)])  # ((weight, reward), (weight, reward))
parser.add_argument('--better_reward',      type=literal,    default = [(1, 0), (1, 10)])
parser.add_argument('--randomness',         type=float,      default = 0)
parser.add_argument('--random_steps',       type=int,        default = 1)
parser.add_argument('--step_cost',          type=float,      default = .99)
parser.add_argument('--body_size',          type=float,      default = 2)    
parser.add_argument('--image_size',         type=int,        default = 8)
parser.add_argument('--max_yaw_change',     type=float,      default = pi/2)
parser.add_argument('--min_speed',          type=float,      default = 0)
parser.add_argument('--max_speed',          type=float,      default = 100)
parser.add_argument('--steps_per_step',     type=int,        default = 5)
parser.add_argument('--speed_scalar',       type=float,      default = .0001)

# Module 
parser.add_argument('--hidden_size',        type=int,        default = 32)   
parser.add_argument('--state_size',         type=int,        default = 32)
parser.add_argument('--actor_hq',           type=literal,    default = True)
parser.add_argument('--critic_hq',          type=literal,    default = False)
parser.add_argument('--forward_lr',         type=float,      default = .01)
parser.add_argument('--alpha_lr',           type=float,      default = .01) 
parser.add_argument('--actor_lr',           type=float,      default = .01)
parser.add_argument('--critic_lr',          type=float,      default = .01)
parser.add_argument('--action_prior',       type=str,        default = "normal")
parser.add_argument("--tau",                type=float,      default = 1)      # For soft-updating target critics

# Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20))
parser.add_argument('--std_max',            type=int,        default = exp(2))
parser.add_argument("--beta",               type=float,      default = 0)

# Entropy
parser.add_argument("--alpha",              type=literal,    default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float,      default = -2)       # Soft-Actor-Critic entropy aim

# Curiosity
parser.add_argument("--curiosity",          type=str,        default = "none")     # Which kind of curiosity
parser.add_argument("--naive_eta",          type=float,      default = 1)        # Scale curiosity
parser.add_argument("--aware_eta",           type=float,     default = 1)        # Scale curiosity
parser.add_argument("--dkl_max",            type=float,      default = 1)        

# Memory buffer
parser.add_argument('--capacity',           type=int,        default = 250)

# Training
parser.add_argument('--epochs',             type=literal,    default = [1000])
parser.add_argument('--steps_per_epoch',    type=int,        default = 25)
parser.add_argument('--batch_size',         type=int,        default = 128)
parser.add_argument('--elbo_num',           type=int,        default = 1)
parser.add_argument('--GAMMA',              type=float,      default = .9)
parser.add_argument("--d",                  type=int,        default = 2)        # Delay to train actors



default_args = parser.parse_args([])
try:    args    = parser.parse_args()
except: args, _ = parser.parse_known_args()



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    

    
def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)




# %%

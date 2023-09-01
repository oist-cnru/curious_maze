#%% 

import argparse, ast
from math import exp, pi
import torch
from torch import nn 

# Finding device for Torch.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arguments to parse. 
def literal(arg_string): return(ast.literal_eval(arg_string))
parser = argparse.ArgumentParser()

# Meta 
parser.add_argument('--init_seed',          type=float,      default = 777,    
                    help='Initial seed value for random number generation.')
parser.add_argument('--device',             type=str,        default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--steps_per_step',     type=int,        default = 5,
                    help='To avoid phasing through walls, physical simulation carried out with small steps.')

# Maze and Agent Details
parser.add_argument('--maze_list',         type=literal,    default = ['t'],
                    help='List of mazes. Agent trains in each maze based on epochs in epochs parameter.')     
parser.add_argument('--max_steps',          type=int,        default = 25,
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',type=float,      default = -1,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--wall_punishment',    type=float,      default = -1,
                    help='Extrinsic punishment for colliding with walls.')
parser.add_argument('--default_reward',     type=literal,    default = [(1, 1)],
                    help='Extrinsic reward for choosing incorrect exit. Format: [(weight, reward), (weight, reward), ...]') 
parser.add_argument('--better_reward',      type=literal,    default = [(1, 0), (1, 10)],
                    help='Extrinsic reward for choosing correct exit. Format: [(weight, reward), (weight, reward), ...]')
parser.add_argument('--randomness',         type=float,      default = 0,
                    help='Which proportion of blocks are randomly selected to randomly change color.')
parser.add_argument('--random_steps',       type=int,        default = 1,
                    help='How many steps an agent makes between selected blocks randomly change color.')
parser.add_argument('--step_cost',          type=float,      default = .99,
                    help='How much extrinsic rewards to exiting are reduced per step.')
parser.add_argument('--body_size',          type=float,      default = 2,
                    help='How larger is the red rubber duck.')    
parser.add_argument('--image_size',         type=int,        default = 8,
                    help='Agent observation images of size x by x by 4 channels.')
parser.add_argument('--max_yaw_change',     type=float,      default = pi/2,
                    help='How much the agent can turn left or right.')
parser.add_argument('--min_speed',          type=float,      default = 0,
                    help='Agent\'s minimum speed.')
parser.add_argument('--max_speed',          type=float,      default = 100,
                    help='Agent\'s maximum speed.')
parser.add_argument('--speed_scalar',       type=float,      default = .0001,
                    help='How agent training relates prediction-error of speed to prediction-error of image.')

# Modules 
parser.add_argument('--hidden_size',        type=int,        default = 32,
                    help='Parameters in hidden layers.')   
parser.add_argument('--state_size',         type=int,        default = 32,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--actor_hq',           type=literal,    default = True,
                    help='Is the agent\'s actor using the forward model\'s hidden-state?')
parser.add_argument('--critic_hq',          type=literal,    default = False,
                    help='Is the critic\'s actor using the forward model\'s hidden-state?')
parser.add_argument('--forward_lr',         type=float,      default = .01,
                    help='Learning rate for forward model.')
parser.add_argument('--alpha_lr',           type=float,      default = .01,
                    help='Learning rate for alpha value.') 
parser.add_argument('--actor_lr',           type=float,      default = .01,
                    help='Learning rate for actor model.')
parser.add_argument('--critic_lr',          type=float,      default = .01,
                    help='Learning rate for critic model.')
parser.add_argument('--action_prior',       type=str,        default = 'normal',
                    help='The actor can be trained based on normal or uniform distributions.')
parser.add_argument('--tau',                type=float,      default = 1,
                    help='Rate at which target-critics approach critics.')      

# Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',            type=int,        default = exp(2),
                    help='Maximum value for standard deviation.')
parser.add_argument('--beta',               type=float,      default = 0,
                    help='Relative importance of complexity over accuracy.')

# Entropy
parser.add_argument('--alpha',              type=literal,    default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument('--target_entropy',     type=float,      default = -2,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')       

# Curiosity
parser.add_argument('--curiosity',          type=str,        default = 'none',
                    help='Which kind of curiosity: none, naive, or aware.')    
parser.add_argument('--naive_eta',          type=float,      default = 1,
                    help='Nonnegative value, how much to consider naive curiosity.')       
parser.add_argument('--aware_eta',           type=float,     default = 1,
                    help='Nonnegative value, how much to consider aware curiosity.')        
parser.add_argument('--dkl_max',            type=float,      default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for aware curiosity.')        

# Memory buffer
parser.add_argument('--capacity',           type=int,        default = 250,
                    help='How many episodes can the memory buffer contain.')

# Training
parser.add_argument('--epochs',             type=literal,    default = [1000],
                    help='List of how many epochs to train in each maze.')
parser.add_argument('--steps_per_epoch',    type=int,        default = 25,
                    help='How many agent-steps between epochs.')
parser.add_argument('--batch_size',         type=int,        default = 128,
                    help='How many episodes are sampled for each epoch.')
parser.add_argument('--elbo_num',           type=int,        default = 1,
                    help='How many times the forward model will predict the next observation.')
parser.add_argument('--GAMMA',              type=float,      default = .9,
                    help='How heavily the critic considers the future.')
parser.add_argument('--d',                  type=int,        default = 2,
                    help='Delay for training actors.')     



# Parsing arguments.
default_args = parser.parse_args([])
try:    args    = parser.parse_args()
except: args, _ = parser.parse_known_args()



# For initializing parameters of neural networks.
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
# Convolution layers with clamped weights. 
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

# Calculating Kullback-Liebler divergence.
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

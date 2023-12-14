#%% 

from math import log2

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, ConstrainedConv2d
spe_size = 1 ; action_size = 2


# For finding how many episodes and steps are in a batch.
def episodes_steps(this):
    return(this.shape[0], this.shape[1])

# For applying reparameterization trick. 
def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

# For applying reparameterization trick.
def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to('cuda' if std.is_cuda else 'cpu')
    return(mu + e * std)

# For applying convolutional layers to recurrent tensors. 
def rnn_cnn(do_this, to_this):
    episodes = to_this.shape[0] ; steps = to_this.shape[1]
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)



# A model for RGBD and speed input.
class Obs_IN(nn.Module):

    def __init__(self, args):
        super(Obs_IN, self).__init__()  
        
        self.args = args
        
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        n_blocks = int(log2(args.image_size) - 2)
        modules = []
        modules.extend([
            ConstrainedConv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect"),
            nn.PReLU(),
            nn.AvgPool2d(
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1))])
        for i in range(n_blocks):
            modules.extend([
                ConstrainedConv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    padding_mode="reflect"),
                nn.PReLU(),
                nn.AvgPool2d(
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1))])
        self.rgbd_in = nn.Sequential(*modules)
        
        example = self.rgbd_in(example)
        rgbd_latent_size = example.flatten(1).shape[1]
        
        self.rgbd_in_lin = nn.Sequential(
            nn.Linear(rgbd_latent_size, args.hidden_size),
            nn.PReLU())
        
        self.speed_in = nn.Sequential(
            nn.Linear(1, args.hidden_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, speed):
        if(len(rgbd.shape) == 4):   rgbd  = rgbd.unsqueeze(1)
        if(len(speed.shape) == 2):  speed = speed.unsqueeze(1)
        rgbd = (rgbd.permute(0, 1, 4, 2, 3) * 2) - 1
        rgbd = rnn_cnn(self.rgbd_in, rgbd).flatten(2)
        rgbd = self.rgbd_in_lin(rgbd)
        speed = (speed - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        speed = self.speed_in(speed)
        return(torch.cat([rgbd, speed], dim = -1))
    
    
    
# A model for prediction RGBD and speed.
class Obs_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
                
        self.args = args
        self.gen_shape = (4, 2, 2) 
        self.rgbd_out_lin = nn.Sequential(
            nn.Linear(2 * args.hidden_size, self.gen_shape[0] * self.gen_shape[1] * self.gen_shape[2]),
            nn.PReLU())
                
        n_blocks = int(log2(args.image_size))
        modules = []
        for i in range(n_blocks):
            modules.extend([
            ConstrainedConv2d(
                in_channels = self.gen_shape[0] if i == 0 else 16,
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU()])
            if i != n_blocks - 1:
                modules.extend([
                    nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)])
        modules.extend([
            ConstrainedConv2d(
                in_channels = 16,
                out_channels = 4,
                kernel_size = (1,1))])
        self.rgbd_out = nn.Sequential(*modules)
        
        self.spe_out = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, spe_size))
        
        self.apply(init_weights)
        self.to(args.device)
                
    def forward(self, h_w_action):
        episodes, steps = episodes_steps(h_w_action)
        rgbd = self.rgbd_out_lin(h_w_action).view((episodes, steps, self.gen_shape[0], self.gen_shape[1], self.gen_shape[2]))
        rgbd_pred = rnn_cnn(self.rgbd_out, rgbd).permute(0, 1, 3, 4, 2)
        spe_pred  = self.spe_out(h_w_action)
        return(rgbd_pred, spe_pred)
    
    
    
# Model for action input.
class Action_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()
        
        self.args = args 
        
        self.action_in = nn.Sequential(
            nn.Linear(action_size, args.hidden_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, action):
        if(len(action.shape) == 2):   action = action.unsqueeze(1)
        action = self.action_in(action)
        return(action)
    
    
    
# Models for MTRNN which are also GRU. 
class MTRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant, args):
        super(MTRNNCell, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        self.new = 1 / time_constant
        self.old = 1 - self.new

        self.r_x = nn.Sequential(
            nn.Linear(input_size, hidden_size))
        self.r_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size))
        
        self.z_x = nn.Sequential(
            nn.Linear(input_size, hidden_size))
        self.z_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size))
        
        self.n_x = nn.Sequential(
            nn.Linear(input_size, hidden_size))
        self.n_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size))
        
        self.apply(init_weights)
        self.to(args.device)

    def forward(self, x, h):
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        z = torch.sigmoid(self.z_x(x) + self.z_h(h))
        new_h = torch.tanh(self.n_x(x) + r * self.n_h(h))
        new_h = new_h * (1 - z)  + h * z
        new_h = new_h * self.new + h * self.old
        return new_h

class MTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant, args):
        super(MTRNN, self).__init__()
        self.args = args
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant, args)
        self.apply(init_weights)

    def forward(self, input, h):
        episodes, steps = episodes_steps(input)
        outputs = []
        for step in range(steps):  
            h = self.mtrnn_cell(input[:, step], h[:, step])
            outputs.append(h)
        outputs = torch.stack(outputs, dim = 1)
        return outputs[:, -1].unsqueeze(1), outputs
        
        

# Forward model.
class Forward(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
                
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        self.zp_mu = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zp_std = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.zq_mu = nn.Sequential(
            nn.Linear(4 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zq_std = nn.Sequential(
            nn.Linear(4 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.mtrnn = MTRNN(
            input_size = args.state_size,
            hidden_size = args.hidden_size, 
            time_constant = 1,
            args = args)
        
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def p(self, prev_action, hq_m1 = None, episodes = 1):
        if(hq_m1 == None): hq_m1  = torch.zeros(episodes, 1, self.args.hidden_size)
        prev_action = self.action_in(prev_action)
        z_input = torch.cat([hq_m1, prev_action], dim = -1) 
        zp_mu, zp_std = var(z_input, self.zp_mu, self.zp_std, self.args)
        zp = sample(zp_mu, zp_std, self.args.device)
        hp, _ = self.mtrnn(zp, hq_m1) 
        return(zp_mu, zp_std, hp)
    
    def q(self, prev_action, rgbd, speed, hq_m1 = None):
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        if(len(rgbd.shape)   == 4):      rgbd        = rgbd.unsqueeze(1)
        if(len(speed.shape)  == 2):      speed       = speed.unsqueeze(1)
        episodes, steps = episodes_steps(rgbd)
        if(hq_m1 == None):     hq_m1 = torch.zeros(episodes, steps, self.args.hidden_size)
        obs = self.obs_in(rgbd, speed)
        prev_action = self.action_in(prev_action)
        z_input = torch.cat((hq_m1, obs, prev_action), dim=-1)
        zq_mu, zq_std = var(z_input, self.zq_mu, self.zq_std, self.args)        
        zq = sample(zq_mu, zq_std, self.args.device)
        hq, _ = self.mtrnn(zq, hq_m1)
        return(zq_mu, zq_std, hq)
        
    def predict(self, action, h): 
        if(len(action.shape) == 2): action  = action.unsqueeze(1)
        if(len(h.shape) == 2):      h       = h.unsqueeze(1)
        h_w_action = torch.cat([self.action_in(action), h], dim = -1)
        pred_rgbd, pred_speed = self.predict_obs(h_w_action)
        return(pred_rgbd, pred_speed)
    
    def forward(self, prev_action, rgbd, speed):
        episodes, steps = episodes_steps(rgbd)
        zp_mu_list = [] ; zp_std_list = [] ;                                                    
        zq_mu_list = [] ; zq_std_list = [] ; zq_rgbd_pred_list = [] ; zq_speed_pred_list = [] ; hq_list = [torch.zeros(episodes, 1, self.args.hidden_size).to(self.args.device)]
        for step in range(steps-1):
            zp_mu, zp_std, hp = self.p(prev_action[:,step],                              hq_list[-1], episodes = episodes)
            zq_mu, zq_std, hq = self.q(prev_action[:,step], rgbd[:,step], speed[:,step], hq_list[-1])
            zq_rgbd_pred, zq_speed_pred = self.predict(prev_action[:,step+1], hq)
            zp_mu_list.append(zp_mu) ; zp_std.append(zp_std) 
            zq_mu_list.append(zq_mu) ; zq_std.append(zq_std) ; hq_list.append(hq)
            zq_rgbd_pred_list.append(zq_rgbd_pred) ; zq_speed_pred_list.append(zq_speed_pred)
        zp_mu, zp_std, hp = self.p(prev_action[:,step+1],                                  hq_list[-1], episodes = episodes)
        zq_mu, zq_std, hq = self.q(prev_action[:,step+1], rgbd[:,step+1], speed[:,step+1], hq_list[-1])
        zp_mu_list.append(zp_mu) ; zp_std_list.append(zp_std) 
        zq_mu_list.append(zq_mu) ; zq_std_list.append(zq_std)
        hq_list.append(hq_list.pop(0))    
        hq_list = torch.cat([hq_list], dim = 1)
        return(
            (zp_mu_list, zp_std_list), 
            (zq_mu_list, zq_std_list, zq_rgbd_pred_list, zq_speed_pred_list, hq_list))



# Actor model.
class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU())
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, action_size),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, h = None):
        if(h == None): h = torch.zeros(1, 1, self.args.hidden_size)
        x = self.lin(h)
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std, self.args.device)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, None)
    
    

# Critic model.
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        self.h_in = nn.Sequential(
            nn.PReLU())
        
        self.gru = nn.GRU(
            input_size =  3 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, 1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, speed, action, h = None):
        obs = self.obs_in(rgbd, speed)
        action = self.action_in(action)
        h, _ = self.gru(torch.cat((obs, action), dim=-1), h)
        Q = self.lin(self.h_in(h))
        return(Q, h)
    


if __name__ == '__main__':
    
    args = default_args
    
    forward = Forward(args)
    
    print('\n\n')
    print(forward)
    print()
    print(torch_summary(forward, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size), (3, 1, args.hidden_size))))
    


    actor = Actor(args)
    
    print('\n\n')
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.hidden_size))))
    
    
    
    critic = Critic(args)
    
    print('\n\n')
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))

# %%

#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log

from utils import default_args, dkl
from maze import Maze
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Actor_HQ, Critic, Critic_HQ

action_size = 2



class Agent:
    
    def __init__(self, GUI = False, args = default_args):
        
        self.start_time = None
        
        self.GUI = GUI
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.arena_name = self.args.arena_list[0]
        self.maze = Maze(self.arena_name, GUI = self.GUI, args = args)
        
        self.target_entropy = args.target_entropy
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=args.alpha_lr, weight_decay=0) 
                
        self.forward = Forward(args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=args.forward_lr, weight_decay=0)   
                           
        self.actor = Actor_HQ(args) if args.actor_hq else Actor(args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_lr, weight_decay=0)
        self.critic1_target = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.train()
        
        self.memory = RecurrentReplayBuffer(args)
        
        
        
    def training(self):
        while(True):
            cumulative_epochs = 0
            prev_arena_name = self.arena_name
            for j, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): self.arena_name = self.args.arena_list[j] ; break
            if(prev_arena_name != self.arena_name): 
                self.maze.arena.stop()
                self.maze = Maze(self.arena_name, GUI = self.GUI , args = self.args)
            self.training_episode()
            if(self.epochs >= sum(self.args.epochs)): break


                
    def step_in_episode(self, prev_a, h_actor, push):
        with torch.no_grad():
            o, s = self.maze.obs()
            a, _, h_actor = self.actor(o, s, prev_a, h_actor)
            action = torch.flatten(a).tolist()
            r, wall_punishment, spot_name, done, action_name = self.maze.action(action[0], action[1])
            no, ns = self.maze.obs()
            if(push): 
                self.memory.push(o, s, a, r + wall_punishment, no, ns, done, done)
        return(a, h_actor, r + wall_punishment, spot_name, done, action_name)
    
    
    
    def step_in_episode_hq(self, prev_a, h_q_m1, push):
        with torch.no_grad():
            o, s = self.maze.obs()
            _, _, h_q = self.forward(o, s, prev_a, h_q_m1)
            a, _, _ = self.actor(h_q)
            action = torch.flatten(a).tolist()
            r, wall_punishment, spot_name, done, action_name = self.maze.action(action[0], action[1])
            no, ns = self.maze.obs()
            if(push): 
                self.memory.push(o, s, a, r + wall_punishment, no, ns, done, done)
                
        return(a, h_q, r + wall_punishment, spot_name, done, action_name)
    
    

    def training_episode(self, push = True):
        print("Episodes:", self.episodes)
        done = False ; prev_a = torch.zeros((1, 1, 2)) ; cumulative_r = 0
        h = torch.zeros((1, 1, self.args.hidden_size))
        self.maze.begin()
        
        for step in range(self.args.max_steps):
            self.steps += 1
            if(not done):
                prev_a, h, r, spot_name, done, _ = self.step_in_episode_hq(prev_a, h, push) if self.args.actor_hq else self.step_in_episode(prev_a, h, push)
                cumulative_r += r
            if(self.steps % self.args.steps_per_epoch == 0):
                self.epoch(batch_size = self.args.batch_size)
        print("Exit: {}. Rewards: {}.".format(spot_name, r))
        self.episodes += 1
    
    
    
    def epoch(self, batch_size):
                                
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
                
        self.epochs += 1

        rgbd, spe, actions, rewards, dones, masks = batch
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        episodes = rewards.shape[0] ; steps = rewards.shape[1]



        # Train forward
        h_qs = [torch.zeros((episodes, 1, self.args.hidden_size)).to(rgbd.device)]
        zp_mus = []       ; zp_stds = []
        zq_mus = []       ; zq_stds = []
        zq_pred_rgbd = [] ; zq_pred_spe = []
        for step in range(steps):
            (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(rgbd[:, step], spe[:, step], actions[:, step], h_qs[-1])
            (_, zq_preds_rgbd), (_, zq_preds_spe) = self.forward.get_preds(actions[:, step+1], zq_mu, zq_std, h_qs[-1], quantity = self.args.elbo_num)
            zp_mus.append(zp_mu) ; zp_stds.append(zp_std)
            zq_mus.append(zq_mu) ; zq_stds.append(zq_std)
            zq_pred_rgbd.append(torch.cat(zq_preds_rgbd, -1)) ; zq_pred_spe.append(torch.cat(zq_preds_spe, -1))
            h_qs.append(h_q_p1)
        h_qs.append(h_qs.pop(0)) ; h_qs = torch.cat(h_qs, dim = 1) ; next_hqs = h_qs[:, 1:] ; hqs = h_qs[:, :-1]
        zp_mus = torch.cat(zp_mus, dim = 1) ; zp_stds = torch.cat(zp_stds, dim = 1)
        zq_mus = torch.cat(zq_mus, dim = 1) ; zq_stds = torch.cat(zq_stds, dim = 1)
        zq_pred_rgbd = torch.cat(zq_pred_rgbd, dim = 1) ; zq_pred_spe = torch.cat(zq_pred_spe, dim = 1)
        
        next_rgbd_tiled = torch.tile(rgbd[:,1:], (1, 1, 1, 1, self.args.elbo_num))
        next_spe_tiled  = torch.tile(spe[:,1:], (1, 1, self.args.elbo_num))

        image_loss = F.binary_cross_entropy_with_logits(zq_pred_rgbd, next_rgbd_tiled, reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * masks / self.args.elbo_num
        speed_loss = self.args.speed_scalar * F.mse_loss(zq_pred_spe, next_spe_tiled,  reduction = "none").mean(-1).unsqueeze(-1) * masks / self.args.elbo_num
        accuracy_for_naive = image_loss + speed_loss
        accuracy            = accuracy_for_naive.mean()
        complexity_for_aware = dkl(zq_mus, zq_stds, zp_mus, zp_stds).mean(-1).unsqueeze(-1) * masks
        complexity          = self.args.beta * complexity_for_aware.mean()        
                        
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
                        
                        
        
        # Get curiosity                  
        if(self.args.dkl_max != None):
            complexity_for_aware = torch.clamp(complexity_for_aware, min = 0, max = self.args.dkl_max)
        naive_curiosity = self.args.naive_eta * accuracy_for_naive  
        aware_curiosity = self.args.aware_eta * complexity_for_aware
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity
        elif(self.args.curiosity == "aware"): curiosity = aware_curiosity
        else:                                curiosity = torch.zeros(rewards.shape)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
                
        
                
        # Train critics
        with torch.no_grad():
            new_actions, log_pis_next, _ = self.actor(next_hqs) if self.args.actor_hq else self.actor(rgbd[:,1:], spe[:,1:], actions[:,1:])
            Q_target1_next, _ = self.critic1_target(next_hqs, new_actions) if self.args.critic_hq else self.critic1_target(rgbd[:,1:], spe[:,1:], new_actions)
            Q_target2_next, _ = self.critic2_target(next_hqs, new_actions) if self.args.critic_hq else self.critic2_target(rgbd[:,1:], spe[:,1:], new_actions)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        Q_1, _ = self.critic1(hqs.detach(), actions[:,1:]) if self.args.critic_hq else self.critic1(rgbd[:,:-1], spe[:,:-1], actions[:,1:])
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2, _ = self.critic2(hqs.detach(), actions[:,1:]) if self.args.critic_hq else self.critic2(rgbd[:,:-1], spe[:,:-1], actions[:,1:])
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
                        
        
        
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(hqs.detach()) if self.args.actor_hq else self.actor(rgbd[:,:-1], spe[:,:-1], actions[:,:-1])
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       alpha = self.args.alpha
            new_actions, log_pis, _ = self.actor(hqs.detach()) if self.args.actor_hq else self.actor(rgbd[:,:-1], spe[:,:-1], actions[:,:-1])

            if self.args.action_prior == "normal":
                loc = torch.zeros(action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            Q_1, _ = self.critic1(hqs.detach(), new_actions) if self.args.critic_hq else self.critic1(rgbd[:,:-1], spe[:,:-1], new_actions)
            Q_2, _ = self.critic2(hqs.detach(), new_actions) if self.args.critic_hq else self.critic2(rgbd[:,:-1], spe[:,:-1], new_actions)
            Q = torch.min(Q_1, Q_2).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            actor_loss = (alpha * log_pis - policy_prior_log_prrgbd - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
                                
                                
                                
        if(accuracy != None):   accuracy = accuracy.item()
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): 
            critic1_loss = critic1_loss.item()
            critic1_loss = log(critic1_loss) if critic1_loss > 0 else critic1_loss
        if(critic2_loss != None): 
            critic2_loss = critic2_loss.item()
            critic2_loss = log(critic2_loss) if critic2_loss > 0 else critic2_loss
        losses = np.array([[accuracy, complexity, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_curiosity = naive_curiosity.mean().item()
        aware_curiosity = aware_curiosity.mean().item()
        if(aware_curiosity == 0): aware_curiosity = None
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, naive_curiosity, aware_curiosity)
    
    
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.forward.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
        
        
if __name__ == "__main__":
    agent = Agent()
# %%

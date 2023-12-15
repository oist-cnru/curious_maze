#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

from utils import default_args, dkl, detach_list
from maze_runner import Maze_Runner
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Critic

action_size = 2



# Class for an agent. 
class Agent:
    
    def __init__(self, args = default_args):
        
        self.args = args
        self.start_time = None
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.maze_name = self.args.maze_list[0]
        self.maze_runner = Maze_Runner(self.maze_name, args = self.args)
        
        # Alpha value.
        self.target_entropy = self.args.target_entropy 
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr) 
        
        # Forward model.
        self.forward = Forward(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr)

        # Actor model.
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr) 
        
        # Two critics and target critics. 
        self.critic1 = Critic(self.args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr)
        self.critic1_target = Critic(self.args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(self.args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr)
        self.critic2_target = Critic(self.args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize replay buffer and begin training.
        self.memory = RecurrentReplayBuffer(self.args)
        self.train()
        
        
    
    # For training the agent. 
    def training(self):        
        while(True):
            # If finished the epochs of the current maze, move to the next maze.
            cumulative_epochs = 0
            prev_maze_name = self.maze_name
            for j, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): self.maze_name = self.args.maze_list[j] ; break
            if(prev_maze_name != self.maze_name): 
                self.maze_runner.maze.stop()
                self.maze_runner = Maze_Runner(self.maze_name, GUI = self.GUI , args = self.args)
            # Otherwise, train in the current maze.
            self.training_episode()
            if(self.epochs >= sum(self.args.epochs)): break
    
    
    
    # How the agent can choose action based on previous action, actor model's hidden state, and observations. 
    def step_in_episode(self, prev_action, hq_m1):
        with torch.no_grad():
            o, s = self.maze_runner.obs()
            _, _, hp = self.forward.p(prev_action, hq_m1)
            _, _, hq = self.forward.q(prev_action, o, s, hq_m1)
            a, _, _ = self.actor(hq) 
            action = torch.flatten(a).tolist()
            r, wall_punishment, spot_name, done, action_name = self.maze_runner.action(action[0], action[1])
            no, ns = self.maze_runner.obs()
            # Add transition to the replay buffer.
            self.memory.push(o, s, a, r + wall_punishment, no, ns, done, done)
        return(a, hp, hq, r + wall_punishment, spot_name, done, action_name)
            
            
    
    # Perform an entire episode. 
    def training_episode(self):
        done = False ; steps = 0
        # Initialize previous action and hidden state and zeros. 
        prev_action = torch.zeros((1, 1, 2))
        hq = None
        
        # Begin.
        all_r = []
        self.maze_runner.begin()
        # Perform step by step.
        for step in range(self.args.max_steps):
            self.steps += 1 
            if(not done):
                steps += 1
                prev_action, hp, hq, r, spot_name, done, _ = self.step_in_episode(prev_action, hq)
                all_r.append(r)
            # If steps can divide steps_per_epoch, have an epoch. 
            if(self.steps % self.args.steps_per_epoch == 0):
                self.epoch(batch_size = self.args.batch_size)
        print('\nExit {}: {}.\nEpisode\'s total extrinsic reward: {}.'.format(self.episodes, spot_name, round(sum(all_r), 3)))
        self.episodes += 1
    
    
    
    # An epoch training forward model, critics, and actor. 
    def epoch(self, batch_size):
        # Sample batch.        
        batch = self.memory.sample(batch_size)
        # If not enough memory for a batch, stop.
        if(batch == False): return(False)
                
        self.epochs += 1

        # Break batch into observations, actions, extrinsic rewards, dones, and masks. 
        rgbd, spe, actions, rewards, dones, masks = batch
        # Add first previous action, zeros. 
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape).to(self.args.device), actions], dim = 1)    
        # Add ones for complete mask. 
        all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1).to(self.args.device), masks], dim = 1)   
        
 
        
        # Train forward model.
        (zp_mus, zp_stds), \
        (zq_mus, zq_stds, pred_rgbd, pred_spe, hqs) = self.forward(actions, rgbd, spe)
        full_hs = hqs
        hs = hqs[:,1:]

        # Accuracy error for predicting images and speeds.
        image_loss = F.binary_cross_entropy_with_logits(pred_rgbd, rgbd[:,1:], reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * masks
        speed_loss = self.args.speed_scalar * F.mse_loss(pred_spe, spe[:,1:],  reduction = "none").mean(-1).unsqueeze(-1) * masks
        # Accuracies for prediciton error curiosity.
        accuracy_for_prediction_error = image_loss + speed_loss
        # Total accuracy for entire batch.
        accuracy = accuracy_for_prediction_error.mean()
        
        # Complexity value for every transition in the batch. 
        complexity_for_hidden_state = [dkl(zq_mu, zq_std, zp_mu, zp_std).mean(-1).unsqueeze(-1) * all_masks for (zq_mu, zq_std, zp_mu, zp_std) in zip(zq_mus, zq_stds, zp_mus, zp_stds)]
        # Total complexity for entire batch.
        complexity = sum([self.args.beta * complexity_for_hidden_state[layer].mean() for layer in range(1)])       
        # Remove first step's complexity, as it's not used for curiosity.
        complexity_for_hidden_state = [layer[:,1:] for layer in complexity_for_hidden_state] 
                        
        # Train forward model based on accuracy and complexity. 
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
                        
                        
        
        # Get curiosity values. 
        # If chosen, clamp complexity for hidden_state curiosity. 
        if(self.args.dkl_max != None):
            complexity_for_hidden_state = [torch.clamp(c, min = 0, max = self.args.dkl_max) for c in complexity_for_hidden_state]
        prediction_error_curiosity = accuracy_for_prediction_error * (self.args.prediction_error_eta)
        hidden_state_curiosities = [complexity_for_hidden_state[layer] * (self.args.hidden_state_eta) for layer in range(1)]
        hidden_state_curiosity = sum(hidden_state_curiosities)
        # If curiosity is actually employed, select it. Otherwise, use zeros.
        if(self.args.curiosity == "prediction_error"):  curiosity = prediction_error_curiosity
        elif(self.args.curiosity == "hidden_state"):    curiosity = hidden_state_curiosity
        else:                                           curiosity = torch.zeros(rewards.shape).to(self.args.device)
        # Add curiosity value to extrinsic rewards.
        rewards += curiosity
                        
        
                
        # Train critics.
        # First, get target Qs.
        with torch.no_grad():
            # Get actor's new actions. 
            new_actions, log_pis_next, _ = self.actor(full_hs)
            # Get target critic opinions.
            Q_target1_next, _ = self.critic1_target(rgbd, spe, new_actions)
            Q_target2_next, _ = self.critic2_target(rgbd, spe, new_actions)
            # Choose more pessimistic opinion.
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            # Ignore first step.
            log_pis_next = log_pis_next[:,1:]
            Q_target_next = Q_target_next[:,1:]
            # Add extrinsic rewards, curiosity, and entropy value. 
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        # Train both critics to predict target Qs. 
        Q_1, _ = self.critic1(rgbd[:,:-1], spe[:,:-1], actions[:,1:])
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2, _ = self.critic2(rgbd[:,:-1], spe[:,:-1], actions[:,1:])
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        # Update target critics.
        self.soft_update(self.critic1, self.critic1_target, self.args.tau)
        self.soft_update(self.critic2, self.critic2_target, self.args.tau)
                        
        
        
        # Train alpha
        if self.args.alpha == None:
            # Get actor's actions.
            _, log_pis, _ = self.actor(hs.detach())
            # Calculate alpha's loss.
            alpha_loss = -(self.log_alpha.to(self.args.device) * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            # Train.
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            # Get new value for alpha. 
            self.alpha = torch.exp(self.log_alpha).to(self.args.device)
        


        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       alpha = self.args.alpha
            # Get actor's actions. 
            new_actions, log_pis, _ = self.actor(hs.detach())
            # Find entropy. 
            if self.args.action_prior == "normal":
                loc = torch.zeros(action_size, dtype=torch.float64).to(self.args.device)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64).to(self.args.device)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            # Get critic's opinions. 
            Q_1, _ = self.critic1(rgbd[:,:-1], spe[:,:-1], new_actions)
            Q_2, _ = self.critic2(rgbd[:,:-1], spe[:,:-1], new_actions)
            # Choose the most pessimistic opinion.
            Q = torch.min(Q_1, Q_2).mean(-1).unsqueeze(-1)
            actor_loss = (alpha * log_pis - policy_prior_log_prrgbd - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            # Train actor to maximize critic appraisal. 
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

    
    
    # How to update target critics. 
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # Change to training mode.
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

#%%
import numpy as np
import pybullet as p
from math import pi, degrees, sin, cos
import torch

from utils import default_args
from maze import Maze



# Class for an agent's interaction with the maze.
class Maze_Runner:
    
    # Initialize
    def __init__(self, maze_name, GUI = True, args = default_args):
        self.args = args
        self.maze = Maze(maze_name, GUI, args)
        self.begin()
        
    # For starting or restarting episodes. 
    def begin(self):
        self.steps = 0 
        self.maze.begin()
        self.agent_pos, self.agent_yaw, self.agent_spe = self.maze.get_pos_yaw_spe()
        
    # Getting an observation: image and speed. 
    def obs(self):
        # Image of red rubber duck's view.
        x, y = cos(self.agent_yaw), sin(self.agent_yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [self.agent_pos[0], self.agent_pos[1], .4], 
            cameraTargetPosition = [self.agent_pos[0] - x, self.agent_pos[1] - y, .4], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.maze.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.maze.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size, height=self.args.image_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.maze.physicsClient)
        rgb = np.divide(rgba[:,:,:-1], 255)
        
        # Adding distance channel to rgb image. 
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d.max() - d)/(d.max()-d.min())
        rgbd = np.concatenate([rgb, d], axis = -1)
        
        # Finish observations.
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        spe = torch.tensor(self.agent_spe).unsqueeze(0).unsqueeze(0)
        return(rgbd, spe)
    
    # Change agent's angle and speed.
    def change_velocity(self, yaw_change, speed):
        old_yaw = self.agent_yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        self.maze.resetBasePositionAndOrientation((self.agent_pos[0], self.agent_pos[1], .5), new_yaw)
        x = -cos(new_yaw)*speed
        y = -sin(new_yaw)*speed
        self.maze.resetBaseVelocity(x, y)
        _, self.agent_yaw, _ = self.maze.get_pos_yaw_spe()
        
    # Given agent's action, implement in maze.
    def action(self, yaw, spe):
        self.steps += 1
        # Adjust curiosity traps. 
        if(self.args.randomness > 0 and self.steps % self.args.random_steps == 0): self.maze.randomize()
        
        # Make agent's action into environment's scale based on minimum and maximum arguments.
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, spe] ; spe.sort() ; spe = spe[1]
        action_name = 'Yaw: {}. Speed: {}.'.format(-1*round(degrees(yaw)), round(spe))
        
        # Implement yaw and speed change. 
        for _ in range(self.args.steps_per_step):
            self.change_velocity(yaw/self.args.steps_per_step, spe/self.args.steps_per_step)
            p.stepSimulation(physicsClientId = self.maze.physicsClient)
            self.agent_pos, self.agent_yaw, self.agent_spe = self.maze.get_pos_yaw_spe()
        
        # Get and finalize extrinsic rewards. 
        end, which, reward = self.maze.exit_reached()
        if(reward > 0): reward *= self.args.step_cost ** self.steps
        col = self.maze.wall_collisions()
        wall_punishment = self.args.wall_punishment if col else 0
        if(not end): end = self.steps >= self.args.max_steps
        exit = which != 'NONE'
        if(end and not exit): reward += self.args.step_lim_punishment

        return(reward, wall_punishment, which, end, action_name)
    
    
    
if __name__ == '__main__':        
    maze = Maze_Runner('t', True, default_args)
# %%

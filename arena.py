#%%
from random import choices, sample
import pandas as pd
import numpy as np
import pybullet as p
import cv2
from math import pi, sin, cos

from utils import default_args

class Exit:
    def __init__(self, name, pos, rew):     
        self.name = name ; self.pos = pos ; self.rew = rew
        
class Arena_Description:
    def __init__(self, start, exits):
        self.start = start 
        self.exit_list = exits
        self.exits = pd.DataFrame(
            data = [[exit.name, exit.pos, exit.rew] for exit in exits],
            columns = ["Name", "Position", "Reward"])
        
        
        
arena_dict = {
    "t.png" : Arena_Description(
        (3, 1),
        [Exit(  "LEFT",     (2,0), "default"),
        Exit(   "RIGHT",    (2,4), "better")]
        ),
    
    "alt.png" : Arena_Description(
        (3, 2),
        [Exit(   "RIGHT",    (2,5), "better")]
        ),
    
    "1.png" : Arena_Description(
        (2,2), 
        [Exit(  "LEFT",    (1,0), "default"),
        Exit(   "RIGHT",    (1,4), "better")]
        ),
    "2.png" : Arena_Description(
        (3,3), 
        [Exit(  "LEFT\nLEFT",   (4,1), "better"),
        Exit(   "LEFT\nRIGHT",  (0,1), "default"),
        Exit(   "RIGHT\nLEFT",  (0,5), "default"),
        Exit(   "RIGHT\nRIGHT", (4,5), "default")]
        ),
    "3.png" : Arena_Description(
        (4,4), 
        [Exit(  "LEFT\nLEFT\nLEFT",    (6,3), "default"),
        Exit(   "LEFT\nLEFT\nRIGHT",   (6,1), "default"),
        Exit(   "LEFt\nRIGHT\nLEFT",   (0,1), "default"),
        Exit(   "LEFT\nRIGHT\nRIGHT",  (0,3), "default"),
        Exit(   "RIGHT\nLEFT\nLEFT",   (0,5), "better"),
        Exit(   "RIGHT\nLEFT\nRIGHT",  (0,7), "default"),
        Exit(   "RIGHT\nRIGHT\nLEFT",  (6,7), "default"),
        Exit(   "RIGHT\nRIGHT\nRIGHT", (6,5), "default")]
        )}



def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2-.5,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data/")
    return(physicsClient)



class Arena():
    def __init__(self, arena_name, GUI = False, args = default_args):
        self.args = args
        if(not arena_name.endswith(".png")): arena_name += ".png"
        self.start = arena_dict[arena_name].start
        self.exits = arena_dict[arena_name].exits
        arena_map = cv2.imread("arenas/" + arena_name)
        w, h, _ = arena_map.shape
        self.physicsClient = get_physics(GUI, w, h)

        self.ends = {} ; self.colors = {} ; cube_locs = []
        for loc in ((x, y) for x in range(w) for y in range(h)):
            pos = [loc[0], loc[1], .5]
            if ((arena_map[loc] == [255]).all()):
                if (not self.exits.loc[self.exits["Position"] == loc].empty):
                    row = self.exits.loc[self.exits["Position"] == loc]
                    end_pos = ((pos[0] - .5, pos[0] + .5), (pos[1] - .5, pos[1] + .5))
                    self.ends[row["Name"].values[0]] = (end_pos, row["Reward"].values[0])
            else:
                ors = p.getQuaternionFromEuler([0, 0, 0])
                color = arena_map[loc][::-1] / 255
                color = np.append(color, 1)
                cube = p.loadURDF("cube.urdf", (pos[0], pos[1], pos[2]), ors, 
                                  useFixedBase=True, physicsClientId=self.physicsClient)
                self.colors[cube] = color
                cube_locs.append(loc)
            
        self.random_pos = sample(cube_locs, k=int(len(cube_locs) * args.randomness))
        
        for cube, color in self.colors.items():
            p.changeVisualShape(cube, -1, rgbaColor = color, physicsClientId = self.physicsClient)

        inherent_roll = pi/2
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        pos = (self.start[0], self.start[1], .5)
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        self.body_num = p.loadURDF("duck.urdf", pos, orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        p.changeVisualShape(self.body_num, -1, rgbaColor = [1,0,0,1], physicsClientId = self.physicsClient)
        
    def begin(self):
        yaw = 0
        spe = self.args.min_speed
        pos = (self.start[0], self.start[1], .5)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        self.resetBasePositionAndOrientation(pos, yaw)
        if(self.args.randomness > 0): self.randomize()
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def resetBasePositionAndOrientation(self, pos, yaw):
        inherent_roll = pi/2
        inherent_pitch = 0
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
        
    def resetBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)
    
    def pos_in_box(self, box):
        (min_x, max_x), (min_y, max_y) = box 
        pos, _, _ = self.get_pos_yaw_spe()
        in_x = pos[0] >= min_x and pos[0] <= max_x 
        in_y = pos[1] >= min_y and pos[1] <= max_y 
        return(in_x and in_y)
    
    def end_collisions(self):
        col = False
        which = "NONE"
        reward = ((1, 0),)
        for end_name, (end, end_reward) in self.ends.items():
            if self.pos_in_box(end):
                col = True
                which = end_name
                reward = self.args.better_reward if end_reward == "better" else self.args.default_reward
        weights = [w for w, r in reward]
        rewards = [r for w, r in reward]
        reward = choices(rewards, weights = weights, k = 1)[0]
        return(col, which, reward)
    
    def other_collisions(self):
        col = False
        for cube in self.colors.keys():
            if 0 < len(p.getContactPoints(self.body_num, cube, physicsClientId = self.physicsClient)):
                col = True
        return(col)
                
    def randomize(self):
        for cube in self.colors.keys():
            pos, _ = p.getBasePositionAndOrientation(cube, physicsClientId = self.physicsClient)
            if(pos[:-1] in self.random_pos):
                p.changeVisualShape(cube, -1, rgbaColor = [choices([0,1])[0], choices([0,1])[0], choices([0,1])[0], 1], physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
        
        
if __name__ == "__main__":
    arena = Arena("t", True)

# %%

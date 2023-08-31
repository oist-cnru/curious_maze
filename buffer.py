#%%

import numpy as np
import torch

from utils import default_args
        


# Ted got this from: Recurrent Off-policy Baselines for Memory-based Continuous Control, https://github.com/zhihanyang2022/off-policy-continuous-control

        

def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)

def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to("cpu")

class RecurrentReplayBuffer:

    """Use this version when num_bptt == max_episode_len"""

    def __init__(
        self, args = default_args, segment_len=None  # for non-overlapping truncated bptt, maybe need a large batch size
    ):
    
        self.args = args
            
        # pointers
      
        self.index = 1
        self.episode_ptr = 0
        self.time_ptr = 0
      
        # trackers
      
        self.starting_new_episode = True
        self.num_episodes = 0
      
        # hyper-parameters
      
        self.capacity = self.args.capacity
        self.o_dim = (self.args.image_size, self.args.image_size, 4)
        self.a_dim = 2
      
        self.max_episode_len = args.max_steps + 1
      
        if segment_len is not None:
            assert self.max_episode_len % segment_len == 0  # e.g., if max_episode_len = 1000, then segment_len = 100 is ok
      
        self.segment_len = segment_len
      
        # placeholders

        self.o = np.zeros((self.args.capacity, self.max_episode_len + 1) + self.o_dim, dtype='float32')
        self.s = np.zeros((self.args.capacity, self.max_episode_len + 1, 1), dtype='float32')
        self.a = np.zeros((self.args.capacity, self.max_episode_len, self.a_dim), dtype='float32')
        self.r = np.zeros((self.args.capacity, self.max_episode_len, 1), dtype='float32')
        self.d = np.zeros((self.args.capacity, self.max_episode_len, 1), dtype='float32')
        self.m = np.zeros((self.args.capacity, self.max_episode_len, 1), dtype='float32')
        
        self.ep_len = np.zeros((self.args.capacity,), dtype='float32')
        self.ready_for_sampling = np.zeros((self.args.capacity,), dtype='int')

    def push(self, o, s, a, r, no, ns, d, cutoff):

        # zero-out current slot at the beginning of an episode

        if self.starting_new_episode:

            self.o[self.episode_ptr] = 0
            self.s[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0

            self.starting_new_episode = False

        # fill placeholders

        self.o[self.episode_ptr, self.time_ptr] = o
        self.s[self.episode_ptr, self.time_ptr] = s
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # fill placeholders

            self.o[self.episode_ptr, self.time_ptr+1] = no
            self.s[self.episode_ptr, self.time_ptr+1] = ns
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # update trackers

            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

        else:

            # update pointers

            self.time_ptr += 1

    def sample(self, batch_size):

        if(self.num_episodes == 0): return(False)
        if(self.num_episodes < batch_size): return self.sample(self.num_episodes)

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=batch_size)

        ep_lens_of_choices = self.ep_len[choices]

        if self.segment_len is None:

            # grab the corresponding numpy array
            # and save computational effort for lstm

            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))

            o = self.o[choices][:, :max_ep_len_in_batch+1, :]
            s = self.s[choices][:, :max_ep_len_in_batch+1, :]
            a = self.a[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]

            # convert to tensors on the right device

            o = as_tensor_on_device(o).view((batch_size, max_ep_len_in_batch+1) + self.o_dim)
            s = as_tensor_on_device(s).view(batch_size, max_ep_len_in_batch+1, 1)
            a = as_tensor_on_device(a).view(batch_size, max_ep_len_in_batch, self.a_dim)
            r = as_tensor_on_device(r).view(batch_size, max_ep_len_in_batch, 1)
            d = as_tensor_on_device(d).view(batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m).view(batch_size, max_ep_len_in_batch, 1)

            return((o, s, a, r, d, m))

        else:

            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)

            o = self.o[choices]
            s = self.s[choices]
            a = self.a[choices]
            r = self.r[choices]
            d = self.d[choices]
            m = self.m[choices]

            o_seg = np.zeros((batch_size, self.segment_len + 1) + self.o_dim)
            s_seg = np.zeros((batch_size, self.segment_len + 1, 1))
            a_seg = np.zeros((batch_size, self.segment_len, self.a_dim))
            r_seg = np.zeros((batch_size, self.segment_len, 1))
            d_seg = np.zeros((batch_size, self.segment_len, 1))
            m_seg = np.zeros((batch_size, self.segment_len, 1))

            for i in range(batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i]) * self.segment_len
                o_seg[i] = o[i][start_idx:start_idx + self.segment_len + 1]
                s_seg[i] = s[i][start_idx:start_idx + self.segment_len + 1]
                a_seg[i] = a[i][start_idx:start_idx + self.segment_len]
                r_seg[i] = r[i][start_idx:start_idx + self.segment_len]
                d_seg[i] = d[i][start_idx:start_idx + self.segment_len]
                m_seg[i] = m[i][start_idx:start_idx + self.segment_len]

            o_seg = as_tensor_on_device(o_seg)
            s_seg = as_tensor_on_device(s_seg)
            a_seg = as_tensor_on_device(a_seg)
            r_seg = as_tensor_on_device(r_seg)
            d_seg = as_tensor_on_device(d_seg)
            m_seg = as_tensor_on_device(m_seg)

            return((o_seg, s_seg, a_seg, r_seg, d_seg, m_seg))

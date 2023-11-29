import torch
from torch import nn

class WalkerPolicy(nn.Module):
    def __init__(self, state_dim=21, action_dim=4):
        self.load_weights()  # call learned stored network weights

    # TODO: implement a determine_actions() function mapping from (N, state_dim) states into (N, action_dim) actions

    def save_weights(self, path='weights_path.t'):
        torch.save(self.state_dict(), path)

    def load_weights(self, path='weights_path.t'):
        self.load_state_dict(torch.load(path))

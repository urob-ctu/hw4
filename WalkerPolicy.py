import torch
from torch import nn

class WalkerPolicy(nn.Module):
    def __init__(self, state_dim=29, action_dim=8):
        super().__init__()
        # self.load_weights()  # load learned stored network weights after initialization

    # TODO: implement a determine_actions() function mapping from (N, state_dim) states into (N, action_dim) actions

    def save_weights(self, path='walker_weights.pt'):
        # helper function to save your network weights
        torch.save(self.state_dict(), path)

    def load_weights(self, path='walker_weights.pt'):
        # helper function to load your network weights
        self.load_state_dict(torch.load(path))

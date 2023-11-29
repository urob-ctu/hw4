import matplotlib.pyplot as plt
import torch

def plot_action_vs_angle(pi):
    test_states = torch.zeros(100, 4)
    test_states[:, 1] = torch.linspace(-torch.pi, torch.pi, 100)

    test_actions = pi.determine_actions(test_states).detach().numpy()
    plt.plot(test_states[:, 1], test_actions[:, 0])
    plt.xlabel('Angle [rad]')
    plt.ylabel('Force')
    plt.grid()
    plt.show()


def plot_training(rewards, p_losses, v_losses=None):
    num_plots = 2 if v_losses is None else 3

    plt.subplot(num_plots, 1, 1)
    plt.plot(rewards, label='mean rewards', color='green')
    plt.ylabel('Mean reward')
    plt.subplot(num_plots, 1, 2)
    plt.plot(p_losses, label='policy loss', color='red')
    plt.ylabel('Policy loss')
    if v_losses is not None:
        plt.subplot(num_plots, 1, 3)
        plt.plot(v_losses, label='value loss', color='blue')
        plt.ylabel('Value loss')
    plt.xlabel('Epoch')
    plt.show()
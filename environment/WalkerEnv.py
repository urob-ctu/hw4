from gymnasium import utils
from .mujoco_env_custom import extendedEnv
from .model_generation import mjcf_to_mjmodel, make_walker_sim
from gymnasium.spaces import Box
import numpy as np
from copy import copy

def default_reward(state, action):
    """reward function for the walker environment, state is (21,) vector, action is (4,) vector"""
    pos = state[:11]  # first 11 elements of state vector are generalized coordinates [xyz, quat, joint_angles]
    vel = state[11:]  # last 10 elements of state vector are generalized velocities [xyz, omega, joint_velocities]
    return vel[0]  # return the x velocity as the reward by default

base_config = {
    'N': 1,
    'vis': 1,
    'reward_fcn' : default_reward
}

class WalkerEnv(extendedEnv, utils.EzPickle):

    def __init__(self, config, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.width = config.get('width', 64)
        self.height = config.get('height', 64)

        # toggle visualization window
        if config.get('vis', 1):
            self.render_mode = 'human'
        else:
            self.render_mode = None
        self.window_title = config.get('window_title', 'mujoco')

        # get generate environment parameters
        self.num_walkers = config.get('N', 1)
        self.skip_steps = 1
        self.frequency = 50

        self.num_states = 21
        self.num_actions = 4

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_states, ), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32, seed=self.np_random)
        model = mjcf_to_mjmodel(make_walker_sim(self.num_walkers, self.frequency))  # create a mujoco model

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": self.frequency//self.skip_steps,
        }

        extendedEnv.__init__(
            self,
            model,
            frame_skip=self.skip_steps,
            render_mode=self.render_mode,
            observation_space=self.observation_space,
            width=self.width,
            height=self.height
        )

        self.step_counter = 0
        self.prev_actions = np.zeros((self.num_walkers, 1))

        self.reward_function = config.get('reward_fcn', default_reward)
        print('Environment ready')

    def vector_step(self, actions):
        ctrl = np.clip(np.array(actions).ravel(), -1, 1)
        self.do_simulation(ctrl, self.frame_skip)  # run the simulation and update states
        self.step_counter += 1
        states = []
        rewards = []

        for i in range(self.num_walkers):
            pos_off, vel_off = 11, 10  # number of positions, velocities per walker
            pos = copy(self.data.qpos[i*pos_off:(i+1)*pos_off])
            pos[0] -= self.init_qpos[i*pos_off] # compensate for the walker's initial x,y position
            pos[1] -= self.init_qpos[i*pos_off + 1]
            vel = self.data.qvel[i*vel_off:(i+1)*vel_off]
            state_i = np.concatenate((pos, vel), dtype=np.float32)
            states.append(state_i)

            rew = self.reward_function(state_i, ctrl[i*4:(i+1)*4])
            rewards.append(rew)  # reward is the negative sin of the angle

        if self.render_mode == 'human':  # if rendering is enabled, render after each simulation step
            self.render()

        self.prev_actions = ctrl.reshape((self.num_walkers, 4))

        return states, rewards

    def _get_obs(self):
        states = []
        for i in range(self.num_walkers):
            pos_off, vel_off = 11, 10
            pos = copy(self.data.qpos[i*pos_off:(i+1)*pos_off])
            pos[0] -= self.init_qpos[i*pos_off]
            pos[1] -= self.init_qpos[i*pos_off + 1]  # subtract starting x,y position
            vel = self.data.qvel[i*vel_off:(i+1)*vel_off]
            state_i = np.concatenate((pos, vel), dtype=np.float32)
            states.append(state_i)
        return states

    def reset_model(self):

        qpos = self.init_qpos  # copy mujoco state vector
        qvel = self.init_qvel

        self.set_state(qpos, qvel)  # set the mujoco state
        self.step_counter = 0
        self.prev_actions = np.zeros((self.num_walkers, 1))
        return self._get_obs()

    def vector_reset(self, seeds=None, options=None):
        """reset all the drones"""
        obs = self.reset_model()
        return obs

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
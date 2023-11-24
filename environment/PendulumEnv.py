from gymnasium import utils
from .mujoco_env_custom import extendedEnv
from .model_generation import mjcf_to_mjmodel, make_pendulum_sim
from gymnasium.spaces import Box
import numpy as np

base_config = {
    'N': 2,
    'vis': True
}

class PendulumEnv(extendedEnv, utils.EzPickle):

    def __init__(self, config, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.width = config.get('width', 64)
        self.height = config.get('height', 64)

        if config.get('vis', 1):
            self.render_mode = 'human'
        self.window_title = config.get('window_title', 'mujoco')

        # get generate environment parameters
        self.double = config.get('double', False)
        self.hardcore = config.get('hardcore', False)
        self.num_pends = config.get('N', 1)
        self.skip_steps = 1
        self.frequency = 50

        if self.double:
            self.num_states = 6
        else:
            self.num_states = 4

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_states, ), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32, seed=self.np_random)
        model = mjcf_to_mjmodel(make_pendulum_sim(self.num_pends, self.frequency, self.double))  # create a mujoco model

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
        self.prev_actions = np.zeros((self.num_pends, 1))
        print('Environment ready')

    def vector_step(self, actions):
        ctrl = np.clip(np.array(actions).ravel(), -1, 1)
        self.do_simulation(ctrl, self.frame_skip)  # run the simulation and update states
        self.step_counter += 1
        states = []
        rewards = []

        for i in range(self.num_pends):
            offset = 3 if self.double else 2
            pos = self.data.qpos[i*offset:offset*(i+1)]
            vel = self.data.qvel[i*offset:offset*(i+1)]
            state_i = np.concatenate((pos, vel), dtype=np.float32)
            states.append(state_i)

            if self.double:
                angle0 = state_i[1]
                angle1 = state_i[2] + angle0
                height = np.cos(angle0) + np.cos(angle1)
            else:
                height = np.cos(state_i[1])

            rew = height
            # rew = np.cos(state_i[1]) - 0.1*state_i[0]**2 - 0.2*actions[i]**2
            # rew = abs(state_i[1]) < 0.1
            if pos[0] > 0.9 or pos[0] < -0.9:
                rew -= 20
            rewards.append(rew)

        if self.render_mode == 'human':  # if rendering is enabled, render after each simulation step
            self.render()

        self.prev_actions = ctrl.reshape((self.num_pends, 1))

        return states, rewards

    def _get_obs(self):
        states = []
        offset = 3 if self.double else 2
        for i in range(self.num_pends):
            pos = self.data.qpos[i*offset:offset*(i+1)]
            vel = self.data.qvel[i*offset:offset*(i+1)]
            state_i = np.concatenate((pos, vel), dtype=np.float32)
            states.append(state_i)
        return states

    def reset_model(self):

        qpos = self.init_qpos  # copy mujoco state vector
        qvel = self.init_qvel

        offset = 3 if self.double else 2
        if self.hardcore:
            qpos[0::offset] = np.random.rand(self.num_pends)*1 - 0.5
            qpos[1::offset] = -np.pi
        else:
            qpos[0::offset] = np.random.rand(self.num_pends)*0.5 - 0.25
            qpos[1::offset] = np.random.rand(self.num_pends)*0.1 - 0.05

        self.set_state(qpos, qvel)  # set the mujoco state
        self.step_counter = 0
        self.prev_actions = np.zeros((self.num_pends, 1))
        return self._get_obs()

    def vector_reset(self, seeds=None, options=None):
        """reset all the pendulums"""
        obs = self.reset_model()
        return obs

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
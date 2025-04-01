"""
Python interface to the CoinRun shared library using ctypes.

On import, this will attempt to build the shared library.
"""

import os
import atexit
import random
import sys
from ctypes import c_int, c_char_p, c_float, c_bool

import gymnasium
import numpy as np
import numpy.ctypeslib as npct

from coinrun.config import Config

# if the environment is crashing, try using the debug build to get
# a readable stack trace
DEBUG = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

game_versions = {
    'standard':   1000,
    'platform': 1001,
    'maze': 1002,
}

def build():
    """
    Build the CoinRun C++ library if needed.
    """
    dirname = os.path.dirname(__file__)
    if len(dirname):
        make_cmd = "QT_SELECT=5 make -C %s" % dirname
    else:
        make_cmd = "QT_SELECT=5 make"

    r = os.system(make_cmd)
    if r != 0:
        print('coinrun: make failed')
        sys.exit(1)

# Try to build the library
build()

if DEBUG:
    lib_path = '.build-debug/coinrun_cpp_d'
else:
    lib_path = '.build-release/coinrun_cpp'

lib = npct.load_library(lib_path, os.path.dirname(__file__))
lib.init.argtypes = [c_int]
lib.get_NUM_ACTIONS.restype = c_int
lib.get_RES_W.restype = c_int
lib.get_RES_H.restype = c_int
lib.get_VIDEORES.restype = c_int

lib.vec_create.argtypes = [
    c_int,    # game_type
    c_int,    # nenvs
    c_int,    # lump_n
    c_bool,   # want_hires_render
    c_float,  # default_zoom
    ]
lib.vec_create.restype = c_int

lib.vec_close.argtypes = [c_int]

lib.vec_step_async_discrete.argtypes = [c_int, npct.ndpointer(dtype=np.int32, ndim=1)]

lib.initialize_args.argtypes = [npct.ndpointer(dtype=np.int32, ndim=1)]
lib.initialize_set_monitor_dir.argtypes = [c_char_p, c_int]

lib.vec_wait.argtypes = [
    c_int,
    npct.ndpointer(dtype=np.uint8, ndim=4),    # normal rgb
    npct.ndpointer(dtype=np.uint8, ndim=4),    # larger rgb for render()
    npct.ndpointer(dtype=np.float32, ndim=1),  # rew
    npct.ndpointer(dtype=bool, ndim=1),     # done
    ]

already_inited = False

def init_args_and_threads(cpu_count=4,
                          monitor_csv_policy='all',
                          rand_seed=None):
    """
    Perform one-time global init for the CoinRun library.  This must be called
    before creating an instance of CoinRunVecEnv.  You should not
    call this multiple times from the same process.
    """
    os.environ['COINRUN_RESOURCES_PATH'] = os.path.join(SCRIPT_DIR, 'assets')
    is_high_difficulty = Config.HIGH_DIFFICULTY

    if rand_seed is None:
        rand_seed = random.SystemRandom().randint(0, 1000000000)

    int_args = np.array([
        int(is_high_difficulty), 
        Config.NUM_LEVELS, 
        int(Config.PAINT_VEL_INFO), 
        Config.USE_DATA_AUGMENTATION, 
        game_versions[Config.GAME_TYPE], 
        Config.SET_SEED, 
        rand_seed
    ]).astype(np.int32)

    lib.initialize_args(int_args)
    
    # If a directory is set for logging, use it for monitor files
    log_dir = os.environ.get('COINRUN_LOG_DIR', '')
    if log_dir == '':
        log_dir = '.'
    
    lib.initialize_set_monitor_dir(log_dir.encode('utf-8'), 
                                  {'off': 0, 'first_env': 1, 'all': 2}[monitor_csv_policy])

    global already_inited
    if already_inited:
        return

    lib.init(cpu_count)
    already_inited = True

@atexit.register
def shutdown():
    global already_inited
    if not already_inited:
        return
    lib.coinrun_shutdown()

class CoinRunVecEnv(gymnasium.Env):
    """
    This is the CoinRun VecEnv, all CoinRun environments are just instances
    of this class with different values for `game_type`

    `game_type`: int game type corresponding to the game type to create, see `enum GameType` in `coinrun.cpp`
    `num_envs`: number of environments to create in this VecEnv
    `lump_n`: only used when the environment creates `monitor.csv` files
    `default_zoom`: controls how much of the level the agent can see
    `render_mode`: str to determine the rendering mode (None, "human", "rgb_array")
    """
    def __init__(self, game_type, num_envs, lump_n=0, default_zoom=5.0, render_mode=None):
        # Make sure the environment is initialized
        if not already_inited:
            init_args_and_threads()
            
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
        self.reward_range = (-float('inf'), float('inf'))
        self.render_mode = render_mode

        self.NUM_ACTIONS = lib.get_NUM_ACTIONS()
        self.RES_W       = lib.get_RES_W()
        self.RES_H       = lib.get_RES_H()
        self.VIDEORES    = lib.get_VIDEORES()

        self.num_envs = num_envs
        self.buf_rew = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=bool)
        self.buf_rgb   = np.zeros([num_envs, self.RES_H, self.RES_W, 3], dtype=np.uint8)
        self.hires_render = Config.IS_HIGH_RES
        if self.hires_render:
            self.buf_render_rgb = np.zeros([num_envs, self.VIDEORES, self.VIDEORES, 3], dtype=np.uint8)
        else:
            self.buf_render_rgb = np.zeros([1, 1, 1, 1], dtype=np.uint8)

        num_channels = 1 if Config.USE_BLACK_WHITE else 3
        self.observation_space = gymnasium.spaces.Box(0, 255, shape=[self.RES_H, self.RES_W, num_channels], dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(self.NUM_ACTIONS)

        self.handle = lib.vec_create(
            game_versions[game_type],
            self.num_envs,
            lump_n,
            self.hires_render,
            default_zoom)
        self.dummy_info = [{} for _ in range(num_envs)]

    def __del__(self):
        if hasattr(self, 'handle'):
            lib.vec_close(self.handle)
        self.handle = 0

    def close(self):
        lib.vec_close(self.handle)
        self.handle = 0

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: Seed for the random number generator.
            options: Additional options for environment reset.

        Returns:
            observation: Initial observation.
            info: Additional information.
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        
        print("CoinRun ignores resets")
        obs, _, _, _, info = self.step_wait()
        return obs, info

    def get_images(self):
        if self.hires_render:
            return self.buf_render_rgb
        else:
            return self.buf_rgb

    def render(self):
        """
        Render the environment.

        Returns:
            Rendered frame depending on the render_mode.
        """
        if self.render_mode is None:
            return None
        
        images = self.get_images()
        
        if self.render_mode == "rgb_array":
            return images
        elif self.render_mode == "human":
            # Here you would normally use a rendering backend
            # For now we'll just return the images array
            return images
        return None

    def step_async(self, actions):
        assert actions.dtype in [np.int32, np.int64]
        actions = actions.astype(np.int32)
        lib.vec_step_async_discrete(self.handle, actions)

    def step_wait(self):
        self.buf_rew = np.zeros_like(self.buf_rew)
        self.buf_done = np.zeros_like(self.buf_done)

        lib.vec_wait(
            self.handle,
            self.buf_rgb,
            self.buf_render_rgb,
            self.buf_rew,
            self.buf_done)

        obs_frames = self.buf_rgb

        if Config.USE_BLACK_WHITE:
            obs_frames = np.mean(obs_frames, axis=-1).astype(np.uint8)[...,None]

        # In the new API, `done` is split into `terminated` and `truncated`
        # For simplicity, we'll set `truncated` to False for all environments
        terminated = self.buf_done
        truncated = np.zeros_like(terminated)

        return obs_frames, self.buf_rew, terminated, truncated, self.dummy_info

    def step(self, actions):
        """
        Step the environment.

        Args:
            actions: Actions to take in the environment.

        Returns:
            observation: Next observation.
            reward: Reward from the action.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode has been truncated.
            info: Additional information.
        """
        self.step_async(actions)
        return self.step_wait()

def make(env_id, num_envs, **kwargs):
    """
    Factory function to create a CoinRunVecEnv.
    
    Args:
        env_id: The environment ID (standard, platform, maze)
        num_envs: The number of environments to create
        **kwargs: Additional arguments to pass to the CoinRunVecEnv constructor
    
    Returns:
        A CoinRunVecEnv instance
    """
    assert env_id in game_versions, 'cannot find environment "%s", maybe you mean one of %s' % (env_id, list(game_versions.keys()))
    return CoinRunVecEnv(env_id, num_envs, **kwargs)
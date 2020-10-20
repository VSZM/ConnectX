import gym
from random import random
from numpy.random import choice

from kaggle_environments import make

from stable_baselines3.common.callbacks import BaseCallback
from common import get_win_percentages_and_score, serializeAndCompress
import numpy as np

def board_flip(mark, board):
    if mark == 1:
        return board
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j, 0] != 0:
                board[i, j, 0] = board[i, j, 0] % 2 + 1
                return board

class ConnectFourGym():
    def __init__(self, opponent_pool=np.asarray(['random']), distribution='even'):
        self.ks_env = make("connectx", debug=True)
        self.rows = self.ks_env.configuration.rows
        self.columns = self.ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(1, self.rows, self.columns), dtype=np.float)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        self.last_action = -1
        self.iter = 0
        self.opponent_pool = opponent_pool
        self.distribution = distribution
        self.init_env()

    def init_env(self):
        if self.distribution == 'even':
            distribution = [1.0 / len(self.opponent_pool)] * len(self.opponent_pool)
        else:
            distribution = self.distribution
        opponent = choice(self.opponent_pool, 1, p=distribution)[0]
        # self.env = self.ks_env.train([opponent, None])
        if self.iter % 2:
            self.env = self.ks_env.train([None, opponent])
        else:
            self.env = self.ks_env.train([opponent, None])

    def reset(self):
        self.iter += 1
        self.init_env()
        self.obs = self.env.reset()
        self.last_action = -1
        return board_flip(self.obs.mark, np.array(self.obs['board']).reshape(1, self.rows, self.columns) / 2)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return board_flip(self.obs.mark,
                          np.array(self.obs['board']).reshape(1, self.rows, self.columns) / 2), reward, done, _

class SaveBestModelCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, model_basename, save_frequency, test_agents, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.step_counter = 0
        self.best_value = -np.inf
        self.model_basename = model_basename
        self.save_frequency = save_frequency
        self.test_agents = test_agents

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.

        """
        self.step_counter += 1
        if self.step_counter % self.save_frequency == 0:
            def trained_agent(obs, config):
                # Use the best model to select a column
                grid = board_flip(obs.mark, np.array(obs['board']).reshape(6, 7, 1))
                col, _ = self.model.predict(grid, deterministic=True)
                # Check if selected column is valid
                is_valid = (obs['board'][int(col)] == 0)
                # If not valid, select random move.
                if is_valid:
                    return int(col)
                else:
                    return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

            score = sum([get_win_percentages_and_score(trained_agent, test_agent, silent=True) for test_agent in
                         self.test_agents])
            if score > self.best_value:
                self.best_value = score
                print('=' * 80)
                print(f'New best agent found with score {score}! Agent encoded:')
                model_serialized = serializeAndCompress(self.model.get_parameters())
                print(model_serialized)
                with open(self.model_basename + str(self.step_counter) + '.model', 'w') as f:
                    f.write(str(model_serialized))

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
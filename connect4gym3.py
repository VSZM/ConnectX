from random import random

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
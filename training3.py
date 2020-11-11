import gym

import torch as th
from torch import nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_zoo3 import minimax_agent, matrix_agent, rule_based_agent, minmax_4, minmax_3, minmax_2, minmax_1
from agents import agentc1, agentc2, agentc3, agentc5, agentc7, agentc9, agentc11
from common import get_win_percentages_and_score
from connect4gym3 import SaveBestModelCallback, ConnectFourGym

agents = ['random', matrix_agent, rule_based_agent, agentc1, agentc2, agentc3, agentc5]

env = ConnectFourGym(agents)
env

vec_env = DummyVecEnv([lambda: env])
vec_env

class Net(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3 = nn.Linear(384, features_dim)

    def forward(self, x):

        x = F.relu(F.batch_norm(self.conv1(x), running_mean=None, running_var=None, training=True))
        x = F.relu(F.batch_norm(self.conv2(x), running_mean=None, running_var=None, training=True))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)

        return x

policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [64, dict(pi=[32, 16], vf=[32, 16])],
    'features_extractor_class': Net,
}

learner = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs)

eval_callback = SaveBestModelCallback('RDaneelConnect4_', 1000, agents)

learner.learn(total_timesteps=5_000_000, callback=eval_callback)

def testagent(obs, config):
    import numpy as np
    obs = np.array(obs['board']).reshape(1, config.rows, config.columns) / 2
    action, _ = learner.predict(obs)
    return int(action)

get_win_percentages_and_score(agent1=testagent, agent2='random')

agent_path = 'submission.py'

submission_beginning = '''def agent(obs, config):
    import numpy as np
    import torch as th
    from torch import nn as nn
    import torch.nn.functional as F
    from torch import tensor

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc3 = nn.Linear(384, 512)
            self.shared1 = nn.Linear(512, 64)
            self.policy1 = nn.Linear(64, 32)
            self.policy2 = nn.Linear(32, 16)
            self.action = nn.Linear(16, 7)

        def forward(self, x):
            x = F.relu(F.batch_norm(self.conv1(x), running_mean=None, running_var=None, training=True))
            x = F.relu(F.batch_norm(self.conv2(x), running_mean=None, running_var=None, training=True))
            x = nn.Flatten()(x)
            x = F.relu(self.fc3(x))
            x = F.dropout(x)
            x = F.relu(self.shared1(x))
            x = F.relu(self.policy1(x))
            x = F.relu(self.policy2(x))
            x = self.action(x)
            x = x.argmax()
            return x


'''

with open(agent_path, mode='w+') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(submission_beginning)

th.set_printoptions(profile="full")

state_dict = learner.policy.to('cpu').state_dict()
state_dict = {
    'conv1.weight': state_dict['features_extractor.conv1.weight'],
    'conv1.bias': state_dict['features_extractor.conv1.bias'],
    'conv2.weight': state_dict['features_extractor.conv2.weight'],
    'conv2.bias': state_dict['features_extractor.conv2.bias'],
    'fc3.weight': state_dict['features_extractor.fc3.weight'],
    'fc3.bias': state_dict['features_extractor.fc3.bias'],

    'shared1.weight': state_dict['mlp_extractor.shared_net.0.weight'],
    'shared1.bias': state_dict['mlp_extractor.shared_net.0.bias'],

    'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
    'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],
    'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
    'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],

    'action.weight': state_dict['action_net.weight'],
    'action.bias': state_dict['action_net.bias'],
}

with open(agent_path, mode='a') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(f'    state_dict = {state_dict}\n')

submission_ending = '''    model = Net()
    model = model.float()
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    model = model.eval()
    obs = tensor(obs['board']).reshape(1, 1, config.rows, config.columns).float()
    obs = obs / 2
    action = model(obs)
    return int(action)'''

with open(agent_path, mode='a') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(submission_ending)


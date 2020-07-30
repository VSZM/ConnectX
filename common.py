from kaggle_environments import evaluate
import pickle
import zlib
import base64 as b64
import numpy as np


# To learn more about the evaluate() function, check out the documentation here: (insert link here)
def get_win_percentages_and_score(agent1, agent2, n_rounds=100, silent=False):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    if not silent:
        print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
        print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
        print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
        print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
        print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))
    return 3 * outcomes.count([1,-1]) + outcomes.count([0, 0])
    

# https://github.com/hill-a/stable-baselines/issues/372
# https://www.kaggle.com/c/connectx/discussion/128591

def serializeAndCompress(value, verbose=True):
    serializedValue = pickle.dumps(value)
    if verbose:
        print('Lenght of serialized object:', len(serializedValue))
    c_data =  zlib.compress(serializedValue, 9)
    if verbose:
        print('Lenght of compressed and serialized object:', len(c_data))
    return b64.b64encode(c_data)
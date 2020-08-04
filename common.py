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

# https://www.kaggle.com/alenic/serializing-q-table-into-submission-py
def serializeAndCompress(value, verbose=True):
    serializedValue = pickle.dumps(value)
    if verbose:
        print('Lenght of serialized object:', len(serializedValue))
    c_data =  zlib.compress(serializedValue, 9)
    if verbose:
        print('Lenght of compressed and serialized object:', len(c_data))
    return b64.b64encode(c_data)


def decompressAndDeserialize(compressedData):
    d_data_byte = b64.b64decode(compressedData)
    data_byte = zlib.decompress(d_data_byte)
    value = pickle.loads(data_byte)
    return value

# Making sure we are always playing mark 1
def board_flip(mark, board):
    if mark == 1:
        return board

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j, 0] != 0:
                board[i, j, 0] = board[i, j, 0]%2 + 1

    return board
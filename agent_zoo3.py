import random

def matrix_agent(obs, config):
    from random import choice
    import numpy as np

    MARK_A = obs.mark
    MARK_B = 3 - MARK_A

    ROWS = config.rows
    COLS = config.columns
    INAROW = config.inarow

    MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]

    TOP_MASK = int(('1' + '0' * ROWS) * COLS, 2)  # 283691315109952
    GRID_MASK = int(('0' + '1' * ROWS) * COLS, 2)  # 279258638311359

    def get_bitmap(obs):
        board = np.asarray(obs.board).reshape(ROWS, COLS)
        board = np.insert(board, 0, 0, axis=0)
        board = np.flip(np.flipud(board).flatten(order='F'))

        str_bitboard = ''.join(map(str, np.int8(board == MARK_A).tolist()))
        str_bitmask = ''.join(map(str, np.int8(board != 0).tolist()))
        return int(str_bitboard, 2), int(str_bitmask, 2)

    def is_win(bitboard):
        for direction in [1, 7, 6, 8]:  # | - \ /
            bb = bitboard & (bitboard >> direction)
            if (bb & (bb >> (2 * direction))) != 0:
                return True
        return False

    def play(bitboard, bitmask, c):
        xboard = bitboard ^ bitmask
        xmask = bitmask | (bitmask + (1 << (c * COLS)))
        xboard = xboard ^ xmask
        return xboard, xmask

    def play_to_win(bitboard, bitmask, c):
        xboard, xmask = play(bitboard, bitmask, c)
        if is_win(xboard):
            return True
        return False

    def valid_moves(bitmask):
        moves = []
        for c in MOVE_ORDER:
            xmask = bitmask | (bitmask + (1 << (c * COLS)))
            if (TOP_MASK & xmask) == 0:
                moves.append(c)
        return moves

    def count_ones(bitboard, bitmask):
        zeros = (~bitmask) & GRID_MASK
        count = 0
        for d1 in [1, 7, 6, 8]:  # | - \ /
            d2 = 2 * d1
            d3 = 3 * d1
            bb = ((bitboard & (zeros >> d1) & (zeros >> d2) & (zeros >> d3)) |
                  (zeros & (bitboard >> d1) & (zeros >> d2) & (zeros >> d3)) |
                  (zeros & (zeros >> d1) & (bitboard >> d2) & (zeros >> d3)) |
                  (zeros & (zeros >> d1) & (zeros >> d2) & (bitboard >> d3)))
            count += bin(bb).count('1')
        return count

    def count_twos(bitboard, bitmask):
        zeros = (~bitmask) & GRID_MASK
        count = 0
        for d1 in [1, 7, 6, 8]:  # | - \ /
            d2 = 2 * d1
            d3 = 3 * d1
            bb = ((bitboard & (bitboard >> d1) & (zeros >> d2) & (zeros >> d3)) |
                  (bitboard & (zeros >> d1) & (bitboard >> d2) & (zeros >> d3)) |
                  (bitboard & (zeros >> d1) & (zeros >> d2) & (bitboard >> d3)) |
                  (zeros & (bitboard >> d1) & (bitboard >> d2) & (zeros >> d3)) |
                  (zeros & (bitboard >> d1) & (zeros >> d2) & (bitboard >> d3)) |
                  (zeros & (zeros >> d1) & (bitboard >> d2) & (bitboard >> d3)))
            count += bin(bb).count('1')
        return count

    def count_threes(bitboard, bitmask):
        zeros = (~bitmask) & GRID_MASK
        count = 0
        for d1 in [1, 7, 6, 8]:  # | - \ /
            d2 = 2 * d1
            d3 = 3 * d1
            bb = ((bitboard & (bitboard >> d1) & (bitboard >> d2) & (zeros >> d3)) |
                  (bitboard & (bitboard >> d1) & (zeros >> d2) & (bitboard >> d3)) |
                  (bitboard & (zeros >> d1) & (bitboard >> d2) & (bitboard >> d3)) |
                  (zeros & (bitboard >> d1) & (bitboard >> d2) & (bitboard >> d3)))
            count += bin(bb).count('1')
        return count

    def heuristic(bitboard, bitmask, c):
        aboard, xmask = play(bitboard, bitmask, c)
        a1 = count_ones(aboard, xmask)
        a2 = count_twos(aboard, xmask)
        a3 = count_threes(aboard, xmask)

        bboard, xmask = play(bitboard ^ bitmask, bitmask, c)
        b1 = count_ones(bboard, xmask)
        b2 = count_twos(bboard, xmask)
        b3 = count_threes(bboard, xmask)

        score = 160 * a3 + 160 * b3 + 40 * a2 + 40 * b2 + 10 * a1 + 10 * b1
        return score

    def act(obs):
        bitboard, bitmask = get_bitmap(obs)

        # start in the middle
        if bitmask == 0:
            return COLS // 2

        good_moves = valid_moves(bitmask)

        # play the only option
        if len(good_moves) == 1:
            return good_moves[0]

            # play to win if you can
        for c in good_moves:
            if play_to_win(bitboard, bitmask, c):
                return c

                # avoid setting up a win
        bad_moves = set()
        for c in good_moves:
            xboard, xmask = play(bitboard, bitmask, c)
            xboard_b = xboard ^ xmask
            next_moves = valid_moves(xmask)
            for cx in next_moves:
                if play_to_win(xboard_b, xmask, cx):
                    bad_moves.add(c)
        good_moves = list(set(good_moves) - bad_moves)

        # block a win if you can
        bitboard_b = bitboard ^ bitmask
        for c in good_moves:
            if play_to_win(bitboard_b, bitmask, c):
                return c

                # play a heuristic move if you can

        # RuntimeError: invalid multinomial distribution (encountering probability entry < 0)
        # https://github.com/ray-project/ray/issues/10265#issuecomment-680160606
        best_score = -float('inf')
        # best_score = -1e15

        best_moves = []
        for c in good_moves:
            score = heuristic(bitboard, bitmask, c)

            if score > best_score:
                best_score = score
                best_moves = [c]
            elif score == best_score:
                best_moves.append(c)

        if best_moves:
            return choice(best_moves)

        # random fallback
        return choice(valid_moves(bitmask))

    return act(obs)

def rule_based_agent(observation, configuration):
    from random import choice

    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_vertical_chance(me_or_enemy):
        for i in range(0, 7):
            if observation.board[i + 7 * 5] == me_or_enemy \
                    and observation.board[i + 7 * 4] == me_or_enemy \
                    and observation.board[i + 7 * 3] == me_or_enemy \
                    and observation.board[i + 7 * 2] == 0:
                return i
            elif observation.board[i + 7 * 4] == me_or_enemy \
                    and observation.board[i + 7 * 3] == me_or_enemy \
                    and observation.board[i + 7 * 2] == me_or_enemy \
                    and observation.board[i + 7 * 1] == 0:
                return i
            elif observation.board[i + 7 * 3] == me_or_enemy \
                    and observation.board[i + 7 * 2] == me_or_enemy \
                    and observation.board[i + 7 * 1] == me_or_enemy \
                    and observation.board[i + 7 * 0] == 0:
                return i
        # no chance
        return -99

    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_horizontal_chance(me_or_enemy):
        chance_cell_num = -99
        for i in [0, 7, 14, 21, 28, 35]:
            for j in range(0, 4):
                val_1 = i + j + 0
                val_2 = i + j + 1
                val_3 = i + j + 2
                val_4 = i + j + 3
                if sum([observation.board[val_1] == me_or_enemy, \
                        observation.board[val_2] == me_or_enemy, \
                        observation.board[val_3] == me_or_enemy, \
                        observation.board[val_4] == me_or_enemy]) == 3:
                    for k in [val_1, val_2, val_3, val_4]:
                        if observation.board[k] == 0:
                            chance_cell_num = k
                            # bottom line
                            for l in range(35, 42):
                                if chance_cell_num == l:
                                    return l - 35
                            # others
                            if observation.board[chance_cell_num + 7] != 0:
                                return chance_cell_num % 7
        # no chance
        return -99

    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_slanting_chance(me_or_enemy, lag, cell_list):
        chance_cell_num = -99
        for i in cell_list:
            val_1 = i + lag * 0
            val_2 = i + lag * 1
            val_3 = i + lag * 2
            val_4 = i + lag * 3
            if sum([observation.board[val_1] == me_or_enemy, \
                    observation.board[val_2] == me_or_enemy, \
                    observation.board[val_3] == me_or_enemy, \
                    observation.board[val_4] == me_or_enemy]) == 3:
                for j in [val_1, val_2, val_3, val_4]:
                    if observation.board[j] == 0:
                        chance_cell_num = j
                        # bottom line
                        for k in range(35, 42):
                            if chance_cell_num == k:
                                return k - 35
                        # others
                        if chance_cell_num != -99 \
                                and observation.board[chance_cell_num + 7] != 0:
                            return chance_cell_num % 7
        # no chance
        return -99

    def check_horizontal_first_enemy_chance():
        # enemy's chance
        if observation.board[38] == enemy_num:
            if sum([observation.board[39] == enemy_num, observation.board[40] == enemy_num]) == 1 \
                    and observation.board[37] == 0:
                for i in range(39, 41):
                    if observation.board[i] == 0:
                        return i - 35
            if sum([observation.board[36] == enemy_num, observation.board[37] == enemy_num]) == 1 \
                    and observation.board[39] == 0:
                for i in range(36, 38):
                    if observation.board[i] == 0:
                        return i - 35
        # no chance
        return -99

    def check_first_or_second():
        count = 0
        for i in observation.board:
            if i != 0:
                count += 1
        # first
        if count % 2 != 1:
            my_num = 1
            enemy_num = 2
        # second
        else:
            my_num = 2
            enemy_num = 1
        return my_num, enemy_num

    # check first or second
    my_num, enemy_num = check_first_or_second()

    def check_my_chances():
        # check my virtical chance
        result = check_vertical_chance(my_num)
        if result != -99:
            return result
        # check my horizontal chance
        result = check_horizontal_chance(my_num)
        if result != -99:
            return result
        # check my slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(my_num, 6, [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20])
        if result != -99:
            return result
        # check my slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(my_num, 8, [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17])
        if result != -99:
            return result
        # no chance
        return -99

    def check_enemy_chances():
        # check horizontal first chance
        result = check_horizontal_first_enemy_chance()
        if result != -99:
            return result
        # check enemy's vertical chance
        result = check_vertical_chance(enemy_num)
        if result != -99:
            return result
        # check enemy's horizontal chance
        result = check_horizontal_chance(enemy_num)
        if result != -99:
            return result
        # check enemy's slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(enemy_num, 6, [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20])
        if result != -99:
            return result
        # check enemy's slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(enemy_num, 8, [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17])
        if result != -99:
            return result
        # no chance
        return -99

    if my_num == 1:
        result = check_my_chances()
        if result != -99:
            return result
        result = check_enemy_chances()
        if result != -99:
            return result
    if my_num == 2:
        result = check_enemy_chances()
        if result != -99:
            return result
        result = check_my_chances()
        if result != -99:
            return result

    # select center as priority (3 > 2 > 4 > 1 > 5 > 0 > 6)
    # column 3
    if observation.board[24] != enemy_num \
            and observation.board[17] != enemy_num \
            and observation.board[10] != enemy_num \
            and observation.board[3] == 0:
        return 3
    # column 2
    elif observation.board[23] != enemy_num \
            and observation.board[16] != enemy_num \
            and observation.board[9] != enemy_num \
            and observation.board[2] == 0:
        return 2
    # column 4
    elif observation.board[25] != enemy_num \
            and observation.board[18] != enemy_num \
            and observation.board[11] != enemy_num \
            and observation.board[4] == 0:
        return 4
    # column 1
    elif observation.board[22] != enemy_num \
            and observation.board[15] != enemy_num \
            and observation.board[8] != enemy_num \
            and observation.board[1] == 0:
        return 1
    # column 5
    elif observation.board[26] != enemy_num \
            and observation.board[19] != enemy_num \
            and observation.board[12] != enemy_num \
            and observation.board[5] == 0:
        return 5
    # column 0
    elif observation.board[21] != enemy_num \
            and observation.board[14] != enemy_num \
            and observation.board[7] != enemy_num \
            and observation.board[0] == 0:
        return 0
    # column 6
    elif observation.board[27] != enemy_num \
            and observation.board[20] != enemy_num \
            and observation.board[13] != enemy_num \
            and observation.board[6] == 0:
        return 6
    # random
    else:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

def minimax_agent(obs, config):
    ################################
    # Imports and helper functions #
    ################################
    import numpy as np

    # Calculates score if agent drops piece in selected column
    def score_move(grid, col, mark, config):
        next_grid = drop_piece(grid, col, mark, config)
        score = get_heuristic(next_grid, mark, config)
        return score

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for score_move: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        score = num_fours * 10000 + num_threes * 10 + num_twos - num_twos_opp * 100 - num_threes_opp * 1000
        return score

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    #########################
    # Agent makes selection #
    #########################

    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximize
    result = random.choice(max_cols)
    return result

import random
import numpy as np

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid


# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)


# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows


N_STEPS = 1
minmax_buff = dict()


def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
    num_fours_opp = count_windows(grid, 4, mark % 2 + 1, config)
    score = num_threes - 1e2 * num_threes_opp - 1e4 * num_fours_opp + 1e6 * num_fours
    return score


def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax(next_grid, nsteps - 1, False, mark, config, -np.Inf, np.Inf)
    return score


# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow


# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False


# Minimax implementation
def minimax(node, depth, maximizingPlayer, mark, config, alpha, beta):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark, config)

    # node_lookup = tuple(np.append(depth, node.flatten()))

    # if node_lookup in minmax_buff:
    # return minmax_buff[node_lookup]

    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax(child, depth - 1, False, mark, config, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        # minmax_buff[node_lookup] = value
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark % 2 + 1, config)
            value = min(value, minimax(child, depth - 1, True, mark, config, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break

        # minmax_buff[node_lookup] = value
        return value


def minmax_agent(obs, config, steps=N_STEPS):
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, steps) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

def minmax_1(obs, config):
    return minmax_agent(obs, config, 1)

def minmax_2(obs, config):
    return minmax_agent(obs, config, 2)

def minmax_3(obs, config):
    return minmax_agent(obs, config, 3)

def minmax_4(obs, config):
    return minmax_agent(obs, config, 4)
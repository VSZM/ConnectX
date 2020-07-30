#%%cython
cimport cython
from libc.stdlib cimport malloc
from libc.stdlib cimport rand, RAND_MAX

# Checks for 3 and 4 in a row
cdef int get_score(int[42] grid, int mark, int[42] prev_grid, int depth):

    cdef int t, row, col, summe
    cdef int num_threes = 0
    cdef int num_threes_opp = 0
     
    # horizontal
    for row in range(6):
        col = 0
        while col < 4:
            summe = 0
            for t in range(4):
                summe += grid[row * 7 + col + t]
    
            if summe < 3 and summe >= 0:
                col += 3 - summe
                continue
                
            if summe > -3 and summe < 0:
                col += 3 + summe
                continue
                
            col += 1    
            summe *= mark
            if summe == 3:
                num_threes += 1
                continue
            if summe == -3:
                num_threes_opp += 1
    
    # vertical
    for col in range(7):
        for row in range(3):
            summe = 0
            for t in range(4):
                summe += grid[(row+t) * 7 + col]
            
            if summe == 0:
                break
                
            summe *= mark
            if summe == 3:
                num_threes += 1
                continue
            if summe == -3:
                num_threes_opp += 1

    # positive diagonal
    for row in range(3):
        col = 0
        while col < 4:
            summe = 0
            for t in range(4):
                summe += grid[(row+t) * 7 + col + t]
    
            if summe < 3 and summe >= 0:
                col += 3 - summe
                continue
                
            if summe > -3 and summe < 0:
                col += 3 + summe
                continue
                
            col += 1   
            summe *= mark
            if summe == 3:
                num_threes += 1
                continue
            if summe == -3:
                num_threes_opp += 1

    # negative diagonal
    for row in range(3,6):
        col = 0
        while col < 4:
            summe = 0
            for t in range(4):
                summe += grid[(row-t) * 7 + col + t]
    
            if summe < 3 and summe >= 0:
                col += 3 - summe
                continue
                
            if summe > -3 and summe < 0:
                col += 3 + summe
                continue
                
            col += 1   
            summe *= mark
            if summe == 3:
                num_threes += 1
                continue
            if summe == -3:
                num_threes_opp += 1
                  
    return num_threes - 2 * num_threes_opp # Alternatively weigh opponents higher or lower


# Checks if it is a terminal position, if true it returns the score
cdef int is_terminal_node(int[42] board, int column, int mark, int row, int player_mark, int depth):
    
    cdef int i = 0
    cdef int j = 0
    cdef int col = 0
    
    # To check if board is full
    for col in range(7):
        if board[col] == 0:
            break
        col += 1
    
    # vertical
    if row < 3:
        for i in range(1, 4):
            if board[column + (row+i) * 7] != mark:
                break
            i += 1
    if i == 4:
        if player_mark == mark:
            return 1000 + depth # depth added, so that it chooses the faster option to win
        else:
            return -1000 - depth
    
    # horizontal
    for i in range(1, 4):
        if (column + i) >= 7 or board[column + i + (row) * 7] != mark:
            break
        i += 1
    for j in range(1, 4):
        if (column - j) < 0 or board[column - j + (row) * 7] != mark:
            break
        j += 1
    if (i + j) >= 5:
        if player_mark == mark:
            return 1000 + depth
        else:
            return -1000 - depth
    
    # top left diagonal
    for i in range(1, 4):
        if (column + i) >= 7 or (row + i) >= 6 or board[column + i + (row + i) * 7] != mark:
            break
        i += 1
    for j in range(1, 4):
        if (column - j) < 0 or(row - j) < 0 or board[column - j + (row - j) * 7] != mark:
            break
        j += 1
    if (i + j) >= 5:
        if player_mark == mark:
            return 1000 + depth
        else:
            return -1000 - depth
    
    # top right diagonal
    for i in range(1, 4):
        if (column + i) >= 7 or (row - i) < 0 or board[column + i + (row - i) * 7] != mark:
            break
        i += 1
    for j in range(1, 4):
        if (column - j) < 0 or(row + j) >= 6 or board[column - j + (row + j) * 7] != mark:
            break
        j += 1
    if (i + j) >= 5:
        if player_mark == mark:
            return 1000 + depth
        else:
            return -1000 - depth
    
    if col == 7:
        return 1 # draw
    return 0 # nobody has won so far


# Initial move is scored with minimax
cdef int score_move(int[42] grid, int col, int mark, int nsteps):

    cdef int[42] next_grid = grid
    cdef int row, row2, column
    cdef int[42] child
    
    for row in range(5, -1, -1):
        if next_grid[7 * row + col] == 0:
            next_grid[7 * row + col] = mark # drop mark
            break
    
    if nsteps > 2: # check if there is an obvious move
        is_terminal = is_terminal_node(next_grid, col, mark, row, mark, nsteps-1)
        if is_terminal != 0:
            return is_terminal

        for column in range(7):
            if next_grid[column] != 0:
                continue
            child = next_grid
            for row2 in range(5, -1, -1):
                if child[7 * row2 + column] == 0:
                    child[7 * row2 + column] = mark*(-1)
                    break

            is_terminal = is_terminal_node(child, column, mark*(-1), row2, mark, nsteps-2)
            if is_terminal != 0:
                return is_terminal + (col == column) #added in case the opponent makes a mistake
        
    cdef int alpha = - 10000000
    cdef int beta = 10000000
    return minimax(next_grid, nsteps-1, 0, mark, grid, alpha, beta, col, row)


# Minimax agent with alpha-beta pruning
cdef int minimax(int[42] node, int depth, int maximizingPlayer, int mark, int[42] grid, int alpha, int beta, int column, int newrow):
    
    cdef int is_terminal 
    if maximizingPlayer:
        is_terminal = is_terminal_node(node, column, mark*(-1), newrow, mark, depth)
        if is_terminal != 0:
            return is_terminal
    if maximizingPlayer == 0:
        is_terminal = is_terminal_node(node, column, mark, newrow, mark, depth)
        if is_terminal != 0:
            return is_terminal

    cdef int value, col, row
    cdef int[42] child
    
    if depth == 0:
        return get_score(node, mark, grid, depth)

    if maximizingPlayer:
        value = -1000000
        for col in range(7):
            if node[col] != 0:
                continue
            child = node
            for row in range(5, -1, -1):
                if child[7 * row + col] == 0:
                    child[7 * row + col] = mark 
                    break
            value = max(value, minimax(child, depth-1, 0, mark, grid, alpha, beta, col, row))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = 1000000
        for col in range(7):
            if node[col] != 0:
                continue
            child = node
            for row in range(5, -1, -1):
                if child[7 * row + col] == 0:
                    child[7 * row + col] = mark*(-1)
                    break
            value = min(value, minimax(child, depth-1, 1, mark, grid, alpha, beta, col, row))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
    

# define the agent   
@cython.cdivision(True)
cpdef int agen(list grid, int mark, int N_STEPS):
    
    if mark == 2:
        mark = -1
        
    cdef int num_max = 1
    cdef int col, sc, i
    cdef int maxsc = -1000001
    cdef int[7] score = [-10000, -10000, -10000, -10000, -10000, -10000, -10000]

    cdef int *c_grid
    
    c_grid = <int *>malloc(42*cython.sizeof(int))
    for i in range(42):
        if grid[i] == 2:
            c_grid[i] = -1
            continue
        c_grid[i] = grid[i]
    
    for col in range(7):
        if c_grid[col] == 0:
            sc = score_move(c_grid, col, mark, N_STEPS)
            if sc == maxsc:
                num_max += 1
                
            if sc > maxsc:
                maxsc = sc
                num_max = 1
                
            score[col] = sc
            
    cdef int choice = int(rand()/(RAND_MAX/num_max))
    cdef int indx = 0
    
    #print(score, mark)

    for i in range(7):
        if score[i] == maxsc:
            if choice == indx:
                return i  
            indx += 1
     
    return 0 # shouldn't be necessary
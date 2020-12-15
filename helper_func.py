import copy
import numpy as np

def place_marble(boardstate, placement_location, marble):
    """
    locate: is a tuple (x,y) with the position in the board
            x is the horizontal coordinate (0-5)
            y is the vertical coordinate (0-5)
    """
    new_boardstate = copy.copy(boardstate)
    if new_boardstate[placement_location] !=0:
        print('ERROR', placement_location, 'already occupied.')
        return 'ERROR'
    else:
        new_boardstate[placement_location] = marble

    return new_boardstate

def rotate_quadrant(boardstate, quadrant, rotation = 1):
    new_boardstate = copy.copy(boardstate)
    # quadrant_defs 
    if quadrant == 1:
        c1 = (0,3)
        c2 = (0,3)
    elif quadrant == 2:
        c1 = (0,3)
        c2 = (3,6)
    elif quadrant == 3:
        c1 = (3,6)
        c2 = (0,3)
    elif quadrant == 4:
        c1 = (3,6)
        c2 = (3,6)

    q = new_boardstate[c1[0]:c1[1],c2[0]:c2[1]] # slice out quadrant
    #print(q)
    q = np.rot90(q, rotation)
    #print(q)
    new_boardstate[c1[0]:c1[1],c2[0]:c2[1]] = q
    return new_boardstate

def fullmove(boardstate, placement_location, quadrant, rotation, marble):
    
    new_boardstate = place_marble(boardstate, placement_location, marble)
    new_boardstate = rotate_quadrant(new_boardstate, quadrant, rotation)

    return new_boardstate

def boardstate_to_key(boardstate):
    replace_neg = copy.copy(boardstate)
    replace_neg[np.where(replace_neg < 0)] = 2
    return np.array2string(replace_neg.flatten(),max_line_width = 200, separator = '')

def ideal_state(input_mat):
    matrix_input = copy.copy(input_mat)
    matrix_input[matrix_input < 0] = 2
    encoder_matrix = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30],[31,32,33,34,35,36]])
    equivalent_matrices = []
    for x in [0,1,2,3]:
        rot_mat = np.rot90(matrix_input, x)
        equivalent_matrices.append(rot_mat)
        equivalent_matrices.append(np.fliplr(rot_mat))
        equivalent_matrices.append(np.flipud(rot_mat))
    
    encoded_scores = [np.dot(encoder_matrix.flatten(),equiv_matrix.flatten()) for equiv_matrix in equivalent_matrices]
    
    best_orientation = equivalent_matrices[encoded_scores.index(max(encoded_scores))]
    best_orientation[best_orientation == 2] = -1
    
    return best_orientation

def boardstate_to_ideal_key(boardstate):
    ideal_state_matrix = ideal_state(boardstate)
    return boardstate_to_key(ideal_state_matrix)


def boardstate_to_nn_input(boardstate):
    ''' returns a 72 length array with positions of player 1 in the first 36 and player 2 in the second 36'''
    ideal_boardstate = ideal_state(boardstate)
    flat = copy.copy(ideal_boardstate).flatten()
    flat1 = copy.copy(flat)
    flat1[flat1>0] = 0
    flat1[flat1<0] = 1
    flat[flat<0] = 0

    return np.concatenate([flat,flat1]).reshape(1,72)

def boardstate_to_cnn_input(boardstate):
    ''' returns a 72 length array with positions of player 1 in the first 36 and player 2 in the second 36'''
    ideal_boardstate = ideal_state(boardstate)
    bs1 = copy.copy(boardstate)
    bs2 = copy.copy(boardstate)
    bs1[bs1<0] = 0
    bs2[bs2>0] = 0
    bs2[bs2<0] = 1
    return np.concatenate([bs1,bs2]).reshape(6,6,2)
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:06:06 2021

@author: Nate
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:37:01 2021

CIS479 - Program 2
@author: Nate Pierce UMID 94712233
"""

import random
import copy
import math

maze_data = [[0,0,0,0,0,0,0],
             [0,'#',0,0,'#',0,0],
             [0,0,0,0,0,0,0],
             [0,'#',0,0,'#',0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]

"""
0 is open space
# is an obstacle
"""

# Constants
ROWS = 6
COLS = 7
OBSTACLES = 4
OPEN_SPOTS = (ROWS*COLS)-OBSTACLES
PZtSt = 0.80 # probability of seeing a block when there is a block
POtSt = 0.85 # probability of seeing an open space when there is an open space
LOC = [4,2] # location in reality. The robot is ignorant of this value.


# Returns a list in form [W, N, E, S] with 0 for no sensed barrier (includes uncertainty in sensing) and 1 for sensed
# barrier.
def evidence(maze): #zn is either 1 or 0
    pos_combs = combinations() 
    look = checkSurrounding(LOC) 
    prob_pos_combs = []
    
    # generates a list [W, N, E, S] of sensory data with weight based on uncertainty in sensing for all possible combinations
    # for i in pos_combs:
    #     arr = [] # list for probabilities associated with [W, N, E, S] e.g. [0.85, 0.85, 0.15, 0.2]

    #     for j in i:
    #             # Generate the arr list
    #             if j == 1: # a barrier exists in this direction
    #                 t = random.choices((0.8,0.2), weights=(80,20)) # sensing uncertainty for barrier
    #             if j == 0: # an open spot exists in this direction
    #                 t = random.choices((0.85,0.15), weights=(85,15)) # sensing uncertainty for open spot
    #             t = t[0] # random.choice forces a 'list' type - convert it to an integer for easier handling
    #             arr.append(t)
                    
    #     prob_pos_combs.append(arr)
    # probability_list = []
    
    #generates a list [W, N, E, S] of sensory data with weight based on uncertainty in sensing
    # arr = []
    # for i in look:
    #         if i == 1: # in reality, a barrier exists in this direction
    #             t = random.choices((0.8,0.2), weights=(80,20)) # sensing uncertainty for barrier
    #         if i == 0: # in reality, an open spot exists in this direction
    #             t = random.choices((0.85,0.15), weights=(85,15)) # sensing uncertainty for open spot
    #         t = t[0]
    #         arr.append(t)
    
    # real_prob = math.prod(arr) 
    
    # Given sensory information for a state Sij,t defined here by look, cycle through all possible comibinations of evidence for 
    # the state and produce a list of all P(Zn,t | Sij,t)
    for i in pos_combs:
        arr = []
        for j in i:
            if j == look[j]:
                t = random.choices((0.8,0.2), weights=(80,20)) # sensing uncertainty for barrier
                print('pos combination elem equal to sensed elem')
            else: 
                t = random.choices((0.85,0.15), weights=(85,15)) # sensing uncertainty for open spot
            t = t[0]
            arr.append(t)
        prob_pos_combs.append(arr)
                
        
    probability_list = []
    # multiply the probabilites for each combination of Z and store in new list
    opt = 0
    for i in prob_pos_combs:
        r = 1
        arr = []
        for j in i:
            r = r*j
        arr.append(r)
        
        if r > opt:
            opt = r
            state = i    
        probability_list.append(arr)
        
    # Convert the most probable state given the sensory data and convert it to list of form such as [0,0,0,1]
    pos = [] # list for barrier/no barrier  associated with [W, N, E, S] e.g. [1, 0, 0, 1]
    for k in range(0, len(state)):
        if state[k] == 0.8:
            pos.append(1)
        if state[k] == 0.85:
            pos.append(0)
        if state[k] == 0.2: # false positive on detecting barrier, i.e. open spot 
            pos.append(0)
        if state[k] == 0.15: # false positive on detecting open spot, i.e. barrier
            pos.append(1)
        
    # Sort the list of all 16 possible sensory combinations and choose the most likely (largest number)    
    probability_list.sort(reverse=True) # sort the list of possible probabilities
    PZiSi = probability_list[0] # select the most likely. This is evidence conditional probability, P(Zi|Si).
    
    return PZiSi   
                        
# Generates an array of arrays with all possible combinations of no barrier/barrier states surrounding a given space in the maze
def combinations():
    combs = []
    for W in range(0,2):
        for N in range(0,2):
            for E in range(0,2):
                for S in range(0,2):
                    combs.append((W,N,E,S))
    return combs

# Returns a count of the number of states within the maze that matches the passed state. 
def countSpots(state):
    l = findSpots()
    count = 0
    for i in l:
        for j in i:
            if state == j:
                count += 1
    return count
                    
# Takes maze data and uses actual location to sense surroundings. Returns a list of of integers 0 (did not see barrier)
# and 1 (did see barrier) e.g. [1,0,0,0] barrier detected to west [W, N, E, S]
def checkSurrounding(loc): 
    sense = []
    i = loc[0]
    j = loc[1]
    
    # look west for object
    if j - 1 == -1: # we're outside the maze
        blockWest = 1
        sense.append(blockWest)
    else: # we're within the maze
        if  maze_data[i][j-1] == '#':
            blockWest = 1
            sense.append(blockWest)
        else:
            blockWest = 0
            sense.append(blockWest)
        
    # look north for object
    if i - 1 == -1:  # we're outside the maze
        blockNorth = 1
        sense.append(blockNorth)
    else: # we're within the maze
        if  maze_data[i-1][j] == '#':
            blockNorth = 1
            sense.append(blockNorth)
        else:
            blockNorth = 0
            sense.append(blockNorth)
        
    # look east for object
    if j + 1 == COLS: # # we're outside the maze
        blockEast = 1
        sense.append(blockEast)
    else: # we're within the maze
        if  maze_data[i][j+1] == '#':
            blockEast = 1
            sense.append(blockEast)
        else:
            blockEast = 0
            sense.append(blockEast)
      
    # look south for object    
    if i + 1 == ROWS:  # we're outside the maze
        blockSouth = 1
        sense.append(blockSouth)
    else: # we're within the maze
        if  maze_data[i+1][j] == '#':
            blockSouth = 1
            sense.append(blockSouth)
        else:
            blockSouth = 0
            sense.append(blockSouth)
            
    return sense

# Auxilliary function that finds the real sensory state for each element in the 2D maze [W, N, E, S]
def findSpots():
    arr = []
    for i in range(0, ROWS):
        l = []
        for j in range(0, COLS):
            location = [i, j]
            t = checkSurrounding(location)
            l.append(t)
        arr.append(l)

    return arr
    
# Movement variable takes form [W, N, E, S], where only one element is 1 (direction of movement) and all others are zero. This param
# is taken as an order for the robot - it is provided to us. 
def move(move, loc):
    x = move.index(1) # 
    i = loc[0]
    j = loc[1]
    look = checkSurrounding(loc)
    global LOC
    
    # movement command is west
    if x == 0:
        t = random.choices(('W','N','S'), weights=(80,10,10)) # movement uncertainty
        if t[0] == 'W' and look[0] == 0:
            j = j-1
        elif t[0] == 'N' and look[1] == 0:
            i = i-1
        elif t[0] == 'S' and look[3] == 0:
            i = i+1
        
    # movement command is north
    if x == 1:
        t = random.choices(('N','W','E'), weights=(80,10,10)) # movement uncertainty
        if t[0] == 'N' and look[1] == 0:
            i = i-1
        elif t[0] == 'W' and look[0] == 0:
            j = j-1
        elif t[0] == 'E' and look[2] == 0:
            j = j+1
            
    LOC = [i, j] 
    
    
def transitional():
    pass
    
def filtering(): 
    pass

def prediction():
    pass

def printState(state):
    print() # new line
    for i in state:
        for j in i:
            if j == '#':
                print(' ', j, end="  ")
            else:    
                print('{0:.2f}'.format(j), end=" ")
        print()

# Generate and return the initial state. The prior is contained within each element of state.
def initial_state(maze):
    for i in range(0, ROWS+1):
        for j in range(0, COLS):
            if maze[i][j] == '#':
                continue
            else:
                maze[i][j] = 1/OPEN_SPOTS
    return maze

if __name__ == '__main__':
    maze = copy.deepcopy(maze_data) # maze_data should be immutable - deepcopy allows different locations for maze and maze_data 
    start_prior = 1/OPEN_SPOTS
    loc = LOC
    
    start = initial_state(maze)
    #countSpots([0,0,1,0])
    #move([0,1,0,0], loc)
    evidence(maze)
    
    printState(start)


def evidence(maze): 
    pos_combs = combinations() 
    look = checkSurrounding(LOC) 
    prob_pos_combs = []
    
    # Given sensory information for a state Sij,t defined here by look, cycle through all possible comibinations of evidence for 
    # the state and produce a list of all P(Zn,t | Sij,t)
    
    for k in range(0, ROWS):
        for m in range(0, COLS):
            for i in pos_combs:
                arr = []
                for j in i:
                    if j == look[j]:
                        t = random.choices((0.8,0.2), weights=(80,20)) # sensing uncertainty for barrier
                    else: 
                        t = random.choices((0.85,0.15), weights=(85,15)) # sensing uncertainty for open spot
                    t = t[0]
                    arr.append(t)
                prob_pos_combs.append(arr)
                        
        
    probability_list = []
    # multiply the probabilites for each combination of Z and store in new list
    opt = 0
    for i in prob_pos_combs:
        r = 1
        arr = []
        for j in i:
            r = r*j
        arr.append(r)
        
        if r > opt:
            opt = r
            state = i    
        probability_list.append(arr)
        
    # Convert the most probable state given the sensory data and convert it to list of form such as [0,0,0,1]
    pos = [] # list for barrier/no barrier  associated with [W, N, E, S] e.g. [1, 0, 0, 1]
    for k in range(0, len(state)):
        if state[k] == 0.8:
            pos.append(1)
        if state[k] == 0.85:
            pos.append(0)
        if state[k] == 0.2: # false positive on detecting barrier, i.e. open spot 
            pos.append(0)
        if state[k] == 0.15: # false positive on detecting open spot, i.e. barrier
            pos.append(1)
        
    # Sort the list of all 16 possible sensory combinations and choose the most likely (largest number)    
    probability_list.sort(reverse=True) # sort the list of possible probabilities
    PZiSi = probability_list[0] # select the most likely. This is evidence conditional probability, P(Zi|Si).
    
    return PZiSi, pos

    

    for k in range(0, ROWS):
        for m in range(0, COLS):
            look = checkSurrounding([k,m]) # real sensory data for position k,m in the maze
            for i in pos_combs: 
                arr = []
                n = 0
                for j in i: # j is 1 or 0 - 1 is a barrier, 0 is a free spot
                    if look[n] == 1 and j == 0: # a barrier exists in this direction but possible combination is false positive
                        t = 0.2
                    if look[n] == 1 and j == 1: # barrier exists in this direction and it is positively detected
                        t = 0.8
                    if look[n] == 0 and j == 0: # open spot in direction and matches possible combination spot
                        t = 0.85
                    if look[n] == 0 and j == 1: # false positive for detecting open spot
                        t = 0.15
                    arr.append(t)
                    n += 1
                prod = math.prod(arr) # P(Zw,n,e,s,i | Sij,i) for element [k, m] in maze
                prob_pos_combs.append(prod)
            prob_pos_combs.sort(reverse=True)
            val = prob_pos_combs[0]
            prob_matrix.append(val)
            print(k, m, prob_matrix)
        
        
            arr = []
    # for k in range(0, ROWS):
    #     for m in range(0, COLS):
    #         c = 0
    #         if maze[k][m] != '#':
    #             p = 0
    #             print(k,m)
    #             for i in range(0, ROWS):
    #                 for j in range(0, COLS):
    #                     if trans_maze[i][j] != '#' and filtered_maze[i][j] != '#' and trans_maze[i][j] != 0.0:
    #                         p += trans_maze[i][j]*filtered_maze[i][j]
    #                         print(p)
    #                     else: continue
    #             maze[k][m] = p
    #             arr.append(p)
    # c = sum(arr)
    # print(c)
    
    # Collect the location of non-zero elements in transitional probability matrix
    
# The maze passed into this function contains the element values P(St+1|Zt) determined during the filtering i.e. the transpition probabilities.
# Sum of elements must equal 1.
def prediction(maze, move): 
    f = copy.deepcopy(maze) # Filtered maze. Elements are P(Si|Zi)
    #trans_maze = transitional(maze, move) # Elements are P(Sinext|Si), not needed?
    x = move.index(1) # extract move command
    
    arr = []
    for i in range(0, ROWS):
        for j in range(0, COLS):
            if f[i][j] != '#':
                look = checkSurrounding([i,j])
                p = 0
                a = 0
                b = 0
                c = 0
                d = 0
                e = 0
                
                if x == 0: # command to move west                    
                    if j - 1 != -1 and j + 1 != COLS and i - 1 != -1 and i + 1 != ROWS: # check if index out of range of maze
                        if f[i][j-1] != '#': a = f[i][j-1]*0.8 # west
                        else: 
                            a = 0
                            e += 0.8
                        if f[i-1][j] != '#': b = f[i-1][j]*0.1 # north
                        else: 
                            b = 0
                            e += 0.1
                        if f[i][j+1] != '#': c = f[i][j+1]*0.8 # east
                        else:
                            c = 0
                            e += 0
                        if f[i+1][j] != '#': d = f[i+1][j]*0.1 # south
                        else: 
                            d = 0
                            e += 0.1
                        p = a + b + c + d + e
                        maze[i][j] = p
                        
                if x == 1: # command to move west                    
                    if j - 1 != -1 and j + 1 != COLS and i - 1 != -1 and i + 1 != ROWS: # check if index out of range of maze
                        if f[i][j-1] != '#': a = f[i][j-1]*0.1 # west
                        else: 
                            a = 0
                            e += 0.8
                        if f[i-1][j] != '#': b = f[i-1][j]*0.8 # north
                        else: 
                            b = 0
                            e += 0.1
                        if f[i][j+1] != '#': c = f[i][j+1]*0.1 # east
                        else:
                            c = 0
                            e += 0
                        if f[i+1][j] != '#': d = f[i+1][j]*0.8 # south
                        else: 
                            d = 0
                            e += 0.1
                        p = a + b + c + d + e
                        maze[i][j] = p
                        
                
                
                
    print('Prediction sum', sum(arr))
    print()
    
    # Uses Bayes Rule to filter the evidence determined by evidence() and the prior Sum of elements must be 1.
def filtering(maze, evi): 
    prior_maze = copy.deepcopy(maze) # Priors are the element values in the maze that is passed to the function
    evidence(maze, evi) # update maze with P(Zw,n,e,s,t|St) for each element

    y = [] # used to troubleshoot - check if sum of all elements is 1
    summation = 0
    
    # find the sum total of all probabilities. Used when finding the norm. 
    for i in range(0, ROWS):
        for j in range(0, COLS):
            if maze[i][j] != '#':
                look = checkSurrounding([i,j])
                count = countSpots(look)
                summation += maze[i][j]
    summation = summation/OPEN_SPOTS
    
    # Calculate probability.    
    for i in range(0, ROWS):
        for j in range(0, COLS):
            num = 0
            norm = 0
            if prior_maze[i][j] != '#':
                PZiSi = maze[i][j]
                num = PZiSi*prior_maze[i][j]
                norm = num/summation
                maze[i][j] = norm
                y.append(norm)
    c = sum(y)  
    print('Filtering sum ', c)       
    return maze
    
# Uses Bayes Rule to filter the evidence determined by evidence() and the prior Sum of elements must be 1.
def filtering(maze, evi): 
    prior_maze = copy.deepcopy(maze) # Priors are the element values in the maze that is passed to the function
    evidence(maze, evi) # update maze with P(Zw,n,e,s,t|St) for each element

    y = [] # used to troubleshoot - check if sum of all elements is 1
    summation = 0
    
    # find the sum total of all probabilities. Used when finding the norm. 
    checked = [] # array of checked states
    for i in range(0, ROWS):
        for j in range(0, COLS):
            if maze[i][j] != '#':
                t = np.round(maze[i][j], 4)
                
                if t not in checked:
                    n = countProbs(maze, maze[i][j])
                    summation += maze[i][j]*n
                    checked.append(t)
    
    # Calculate probability.    
    for i in range(0, ROWS):
        for j in range(0, COLS):
            num = 0
            norm = 0
            if prior_maze[i][j] != '#':
                PZiSi = maze[i][j]
                num = PZiSi*prior_maze[i][j]
                norm = num/summation
                maze[i][j] = norm
                y.append(norm)
    c = sum(y)  
    print('Filtering sum ', c)       
    return maze

# Given a maze state, generates a probability matrix depending on the move command.    
def transitional(maze, move):
    global LOC
    x = move.index(1)
    look = checkSurrounding(LOC)
    
    # initialize array of zeroes
    for i in maze:
        c = 0 
        for j in i:
            if j == '#': 
                c += 1 
                continue
            else:
                i[c] = 0.0
                c += 1
    i = LOC[0]
    j = LOC[1]
     
    # Generate transition matrix depending on move command           
    if x == 0: # command to move west
        if look[0] == 1: # barrier exists to the west
            t = [0.0, 0.1, 0.8, 0.1, 0.8] # [W, N, E, S, Stay at Current Location]
            if look[1] == 0: maze[i-1][j] = t[1] # north
            if look[2] == 0: maze[i][j+1] = t[2] # east
            if look[3] == 0: maze[i+1][j] = t[3] # south
            maze[i][j] = t[4] # current location
        else:
            t = [0.8, 0.1, 0.8, 0.1, 0.0] # high probability of moving west
            if look[0] == 0: maze[i][j-1] = t[0] # west
            if look[1] == 0: maze[i-1][j] = t[1] # north
            if look[2] == 0: maze[i][j+1] = t[2] # east
            if look[3] == 0: maze[i+1][j] = t[3] # south
            maze[i][j] = t[4] # current location
            
            
    if x == 1: # command to move north
        if look[1] == 1: # barrier exists to the north
            t = [0.1, 0.0, 0.1, 0.8, 0.8] # [W, N, E, S, Stay at Current Location]
            if look[0] == 0: maze[i][j-1] = t[0] # west
            if look[2] == 0: maze[i][j+1] = t[2] # east
            if look[3] == 0: maze[i+1][j] = t[3] # south
            maze[i][j] = t[4] # current location
        else:
            t = [0.1, 0.8, 0.1, 0.8, 0.0] # high probability of moving north
            if look[0] == 0: maze[i][j-1] = t[0] # west
            if look[1] == 0: maze[i-1][j] = t[1] # north
            if look[2] == 0: maze[i][j+1] = t[2] # east
            if look[3] == 0: maze[i+1][j] = t[3] # south
            maze[i][j] = t[4] # current location
            
    return maze
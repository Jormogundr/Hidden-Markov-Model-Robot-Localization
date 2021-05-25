# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:37:01 2021

CIS479 - Program 2
@author: Nate Pierce UMID 94712233
"""

import copy
import math
import numpy as np

maze_data = [[0,0,0,0,0,0,0],
             [0,'#',0,0,'#',0,0],
             [0,0,0,0,0,0,0],
             [0,'#',0,0,'#',0,0],
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
LOC = [4,2] # location in reality. The robot is ignorant of this value. Mutable. 

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

# Takes the maze and evidence - checks if evidence matches reality and assigns probability accordingly for each element in the maze, i.e.
# the function generates P(Zw,n,e,s,t|St) for every state (element ij in matrix) in the maze. 
def evidence(maze, evi):
    for k in range(0, ROWS):
        for m in range(0, COLS):
            if maze[k][m] != '#':
                look = checkSurrounding([k, m])
                prod = 0
                arr = []
                for n in range(0, len(look)): 
                        if look[n] == evi[n] and look[n] == 0: # an open spot is sensed and there is an open spot
                            t = 0.85
                        if look[n] == evi[n] and look[n] == 1: # a barrier is sensed and there is a barrier
                            t = 0.8
                        if look[n] != evi[n] and look[n] == 0: # a barrier is sensed and there is an open spot
                            t = 0.15
                        if look[n] != evi[n] and look[n] == 1: # an open spot is sensed and there is a barrier
                            t = 0.2
                        arr.append(t)
                        n += 1
                prod = math.prod(arr)
                maze[k][m] = prod
            else: continue
    print()


# The maze passed into this function contains the element values P(St+1|Zt) determined during the filtering i.e. the transpition probabilities.
# Sum of elements must equal 1.
def prediction(maze, move): 
    f = copy.deepcopy(maze) # Filtered maze. Elements are P(Si|Zi)
    x = move.index(1) # extract move command
    arr = []
    for i in range(0, ROWS):
        for j in range(0, COLS):
            if f[i][j] != '#':
                
                # Initialize transition probabilities - must be reset for every element in maze!
                p = 0 # running total
                a = 0 # west
                b = 0 # north
                c = 0 # east
                d = 0 # south
                e = 0 # stationary probability 
                
                if x == 0: # command to move west                    
                    if j - 1 > -1: 
                        if f[i][j-1] != '#': a = f[i][j-1]*0.8 
                        else: e += 0.8
                    else: e += 0.8
                    if i - 1 > -1: 
                        if f[i-1][j] != '#': b = f[i-1][j]*0.1
                        else: e += 0.1
                    else: e += 0.1
                    if j + 1 < COLS:
                        if f[i][j+1] != '#': c = f[i][j+1]*0.8
                        else: e += 0.0
                    else: e += 0.0
                    if i + 1 < ROWS:
                        if f[i+1][j] != '#': d = f[i+1][j]*0.1 
                        else: e += 0.0 
                    else: e += 0.0 
                    e = e*f[i][j]
                    p = a + b + c + d + e
                    maze[i][j] = p
                        
                        
                if x == 1: # command to move north                    
                    if j - 1 > -1: 
                        if f[i][j-1] != '#': a = f[i][j-1]*0.1 
                        else: e += 0.1
                    else: e += 0.1
                    if i - 1 > -1: 
                        if f[i-1][j] != '#': b = f[i-1][j]*0.0
                        else: e += 0.8
                    else: e += 0.8
                    if j + 1 < COLS:
                        if f[i][j+1] != '#': c = f[i][j+1]*0.1
                        else: e += 0.1
                    else: e += 0.1
                    if i + 1 < ROWS:
                        if f[i+1][j] != '#': d = f[i+1][j]*0.8 
                        else: e += 0.0 
                    else: e += 0.0 
                    e = e*f[i][j]
                    p = a + b + c + d + e
                    maze[i][j] = p
                    arr.append(p)
    t = sum(arr)
    print("Prediction sum is ", t)

    
    
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
            
                        
# Generates an array of arrays with all possible combinations of no barrier/barrier states surrounding a given space in the maze
def combinations():
    combs = []
    for W in range(0,2):
        for N in range(0,2):
            for E in range(0,2):
                for S in range(0,2):
                    combs.append((W,N,E,S))
    return combs
                    
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
    
def move(move):
    global LOC
    x = move.index(1) # 
    i = LOC[0]
    j = LOC[1]
    
    # movement command is west
    if x == 0:
        j = j - 1
        
    # movement command is north
    if x == 1:
        i = i - 1
            
    LOC = [i, j] 
            
    
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

# Counts the number of spots in the maze with matching probablities.
def countProbs(maze, prob):
    count = 0
    prob = np.round(prob, 3)
    for i in range(0, ROWS):
        for j in range(0, COLS):
            if maze[i][j] != '#':
                elem = round(maze[i][j], 3)
                if elem == prob:
                    count +=1 
    return count
            

# Returns a count of the number of states within the maze that matches the passed state. 
def countSpots(state):
    l = findSpots()
    count = 0
    for i in l:
        for j in i:
            if state == j:
                count += 1
    return count

# Helper function to convert a state in probability form to state in barrier/no barrier form
def convertState(prob_state):
    for i in prob_state:
        if i == 0.85: i = 0
        if i == 0.8: i = 1
        if i == 0.2: i = 1
        if i == 0.15: i = 0
    return prob_state
            

def printState(state):
    for i in state:
        for j in i:
            if j == '#':
                print('#', end="   ")
            else:    
                j = j*100
                print('{0:.2f}'.format(j), end="   ")
        print()

# Generate and return the initial state. The prior is contained within each element of state.
def initial_state(maze):
    for i in range(0, ROWS):
        for j in range(0, COLS):
            if maze[i][j] == '#':
                continue
            else:
                maze[i][j] = 1/OPEN_SPOTS
    return maze

if __name__ == '__main__':
    maze = copy.deepcopy(maze_data) # maze_data should be immutable - deepcopy allows different locations for maze and maze_data 
    start_prior = 1/OPEN_SPOTS # initialize probability matrix with prior
    move_sequence = [[0,1,0,0], [0,1,0,0], [1,0,0,0], [1,0,0,0]]
    evidence_sequence = [[0,0,0,0], [1,0,0,0], [0,0,0,0], [0,1,0,1]]
    maze = initial_state(maze)
    
    # this block for testing only
    # filtering(maze, [0,1,0,0])
    # filtering(maze, [0,1,0,0])
    # filtering(maze, [1,0,0,0])
    
    print('\nInitial Location Probabilities')
    printState(maze)
    for i in range(0, len(move_sequence)):
        print('Location before move is ', LOC)
        
        # filtering
        print('\nFiltering after evidence ', evidence_sequence[i] )
        filtering(maze, evidence_sequence[i])
        printState(maze) 
        
        # prediction
        print('\nPrediction after Action', move_sequence[i])
        prediction(maze, move_sequence[i])
        printState(maze) 
        
        # move
        move(move_sequence[i])
        
    # last evidence
    print('\nFiltering after evidence [1,0,0,0]')   
    printState(filtering(maze, [1,0,0,0])) 
    
       

    
    


    
        
    
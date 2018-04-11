# ColumbiaX: CSMM.101x Artificial Intelligence (AI) 2018.01
# Project: search algorithms
# Author: Honghu Xu, honghu.xv@icloud.com

import sys
import numpy as np
import math
from enum import Enum
from queue import Queue
import time
import resource
from collections import deque
from queue import PriorityQueue



class Action(Enum):
    
    """
    Define a class to record element movement actions for a grid
    """

    LEFT  =  ( 0, -1)
    RIGHT =  ( 0,  1)
    UP    =  (-1,  0) 
    DOWN  =  ( 1,  0)
    
    @property
    def delta(self):
        return (self.value[0], self.value[1])  




def valid_actions(tuple):
    """
    Returns a list of valid actions given a tuple including 0.
    """
    grid = np.array(tuple).reshape(3,-1)
    
    index = np.where(grid == 0)
    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN]
    
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = int(index[0]),int(index[1])

    
    

    # check if the node is off the grid or it's an obstacle
    if x == 0:
        valid.remove(Action.UP)
    if x == n:
        valid.remove(Action.DOWN)
    if y == 0:
        valid.remove(Action.LEFT)
    if y == m:
        valid.remove(Action.RIGHT)     
    return valid


def hueristic(start,goal):
    """
    Returns a cost: sum of start->goal distance for number 0->8
    """

    start_grid = np.array(start).reshape(3,-1)
    goal_grid  = np.array(goal).reshape(3,-1)


    sum = 0


    for i in range(0,len(goal_grid[0])):
        for j in range(0,len(goal_grid[1])):
            tile = goal_grid[i][j]
            for k in range(0,len(start_grid[0])):
                for l in range(0,len(start_grid[1])):
                    if start_grid[k][l] == tile:
                        sum += (k-i)**2 + (j-l)**2

    return sum



def dfs(start,goal):
    """
    Returns a valid steps list by given the start and goal tuple. implement BFS algorithm.
    """

    path = []
    depth = []

    # initialize variables
    max_depth = 0 

    # frontier variables 
    max_fringe_size = 0

    # initialize a stack object and add the start location to it:

    queue = deque()
    queue.append(start)

    #  initialize a set() object for visited list and add the start location to it
    visited = set()
    visited.add(start)

    # initialize stack object for depth calculation

    depth = deque()
    depth.append(0)


    # define an empty dictionary, where you'll record how you moved through the grid and a goal location,
    branch = {}
    found = False
    
    
    while queue:
        # deque and store the explored node
        # current_node = queue.get()
        current_node = queue.pop()
        visited.add(current_node)
        # dep = depth.get()
        dep = depth.pop()

        
        
        # goal check
        if current_node == goal:
            print('Found the Solution')
            found = True
            break
        else:
            count = 0
            for action in valid_actions(current_node):
                # get movement indicator from actions list
                da = action.delta
                
                # tuple -> grid transformation
                grid = np.array(current_node).reshape(3,-1)
                
                # find grid index of 0
                index = np.where(grid == 0)
                x,y = int(index[0]),int(index[1])
                
                #grid manipulation to exchange 0 and neighbor elements. 
                grid[x+da[0],y+da[1]],grid[x,y] = grid[x,y],grid[x+da[0],y+da[1]]
                
                # grid -> tuple transformation
                next_node = tuple(grid.flatten().tolist())
                

                # Check if the new node has been visited before.
                # If the node has not been visited:
                # 1. Mark it as visited
                # 2. Add it to the queue
                # 3. Add how I got there to branch
                if next_node not in visited:
                    visited.add(next_node)
                    # queue.put(next_node)
                    queue.append(next_node)
                    # depth.put(dep+1)
                    depth.append(dep+1)
                    
                    branch[next_node] = (current_node, action)
                    count += 1

            fringe_size = len(queue)
            if fringe_size > max_fringe_size:
                max_fringe_size = fringe_size

            if count > 0:
                if dep + 1 > max_depth:
                    max_depth = dep + 1


    nodes = 0            
    
    if found:

        nodes = len(branch)
        
        # traceback to find the depth by using of the branch dictionary.
        n = goal
        # print(branch[n][0])
        while branch[n][0] != start:
            
            path.append(branch[n][1])
            n = branch[n][0]
            
        path.append(branch[n][1])

        
    return path[::-1],nodes,max_depth,fringe_size




def ast(hueristic,start,goal):
    """
    Returns a valid steps list by given the start and goal tuple. implement BFS algorithm.
    """
    path = []
    depth = []

    # initialize variables
    max_depth = 0 

    # initialize a Queue() object and add the start location to it:
    # queue  = Queue()

    queue = PriorityQueue()
    # queue.put(start) 
    queue.put((0,start))
    #  initialize a set() object for visited list and add the start location to it
    visited = set()
    visited.add(start)

    # initialize Queue() object for depth calculation
    depth = Queue()
    depth.put(0)


    # define an empty dictionary, where you'll record how you moved through the grid and a goal location,
    branch = {}
    found = False
         
    while not queue.empty():
        # deque and store the explored node
        # current_node = queue.get()
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        visited.add(current_node)
        dep = depth.get()


        if current_node == goal:
            print('Found the Solution')
            found = True
            break
        else:
            for action in valid_actions(current_node):
            # get movement indicator from actions list
                da = action.delta

                # tuple -> grid transformation
                grid = np.array(current_node).reshape(3,-1)

                # find grid index of 0
                index = np.where(grid == 0)
                x,y = int(index[0]),int(index[1])

                #grid manipulation to exchange 0 and neighbor elements. 
                grid[x+da[0],y+da[1]],grid[x,y] = grid[x,y],grid[x+da[0],y+da[1]]

                # grid -> tuple transformation
                next_node = tuple(grid.flatten().tolist())

                # calculate the heuristic cost.
                new_cost = current_cost + hueristic(next_node,goal)


                # Check if the new node has been visited before.
                # If the node has not been visited:
                # 1. Mark it as visited
                # 2. Add it to the queue
                # 3. Add how I got there to branch
                
                if next_node not in visited:
                    visited.add(next_node)
                    # queue.put(next_node)
                    queue.put((new_cost,next_node))

                    depth.put(dep+1)
                    # branch[next_node] = (current_node, action)
                    branch[next_node] = (new_cost,current_node,action)

            if dep + 1 > max_depth:
                max_depth = dep + 1


    path_cost = 0 
    nodes = 0 
                        
    if found:

        # path_cost = 0

        nodes = len(branch)

        # traceback to find the depth by using of the branch dictionary.
        n = goal
        # print(branch[n][0])
        path_cost = branch[n][0]

        # while branch[n][0] != start:
        while branch[n][1] != start:

        # path.append(branch[n][1])
            path.append(branch[n][2])
            # n = branch[n][0]
            n = branch[n][1]

        # path.append(branch[n][1])
        path.append(branch[n][2])

    return path[::-1],max_depth,nodes,path_cost


def bfs(start,goal):
    """
    Returns a valid steps list by given the start and goal tuple. implement BFS algorithm.
    """

    path = []
    depth = []

    # initialize variables
    max_depth = 0 

    # initialize a Queue() object and add the start location to it:
    queue = Queue()
    queue.put(start)
    #  initialize a set() object for visited list and add the start location to it
    visited = set()
    visited.add(start)

    # initialize Queue() object for depth calculation
    depth = Queue()
    depth.put(0)


    # define an empty dictionary, where you'll record how you moved through the grid and a goal location,
    branch = {}
    found = False

    max_fringe_size = 0 
    
    
    while not queue.empty():
        # deque and store the explored node
        current_node = queue.get()
        visited.add(current_node)
        dep = depth.get()
        
        
        # goal check
        if current_node == goal:
            print('Found the Solution')
            found = True
            break
        else:
            for action in valid_actions(current_node):
                # get movement indicator from actions list
                da = action.delta
                
                # tuple -> grid transformation
                grid = np.array(current_node).reshape(3,-1)
                
                # find grid index of 0
                index = np.where(grid == 0)
                x,y = int(index[0]),int(index[1])
                
                #grid manipulation to exchange 0 and neighbor elements. 
                grid[x+da[0],y+da[1]],grid[x,y] = grid[x,y],grid[x+da[0],y+da[1]]
                
                # grid -> tuple transformation
                next_node = tuple(grid.flatten().tolist())
                

                # Check if the new node has been visited before.
                # If the node has not been visited:
                # 1. Mark it as visited
                # 2. Add it to the queue
                # 3. Add how I got there to branch
                if next_node not in visited:
                    visited.add(next_node)
                    queue.put(next_node)
                    depth.put(dep+1)
                    branch[next_node] = (current_node, action)

            fringe_size = queue.qsize()
            if fringe_size > max_fringe_size:
                max_fringe_size = fringe_size

            if dep + 1 > max_depth:
                max_depth = dep + 1

    nodes = 0

    if found:

        nodes = len(branch)
        
        # traceback to find the depth by using of the branch dictionary.
        n = goal
        #print(branch[n][0])
        while branch[n][0] != start:
            
            path.append(branch[n][1])
            n = branch[n][0]
            
        path.append(branch[n][1])

    return path[::-1],nodes,max_depth,max_fringe_size




if __name__ == "__main__":

    method_input = sys.argv[1]

    start = tuple(list(map(int,sys.argv[2].split(','))))

    # start = (5,8,3,2,0,1,4,7,6) 
    goal  = (0,1,2,3,4,5,6,7,8)

    if method_input == "ast":
        t0 = time.time()
        path,depth,node,path_cost = ast(hueristic,start,goal)
        t1 = time.time()
        print("Heuristic Cost: %d" % path_cost)

    if method_input == "bfs":
        t0 = time.time()
        path,node,depth,fringe = bfs(start,goal)
        t1 = time.time()
        print("Fringe Size: %d" % fringe)

    if method_input == "dfs":
        t0 = time.time()
        path,node,depth,fringe = dfs(start,goal)
        t1 = time.time()
        print("Fringe Size: %d" % fringe)


    # simplify the output: e.g. <Action.LEFT: (0, -1)> --> LEFT
    path_output = []
    for i in range(len(path)):
        path_output.append(str(path[i])[7:])


    # output of Max Ram Usage
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


    print("Cost Of Path: %d" % len(path))
    print("Notes Expended: %d" % node)
    print("Search Depth: %d" % len(path))
    print("Max Search Depth: %d" % depth)
    print("Max Ram Usage: %d " % usage)
    print("Running Time(in second): %e" % (t1-t0))
    print("Path To Goal: %r" % path_output)
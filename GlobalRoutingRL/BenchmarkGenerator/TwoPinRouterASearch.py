from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import Initializer as init
import GridGraph as gridgraph
import Router

import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class AStarSearchGraph(object):
    def __init__(self,gridParameters,capacity):
        self.capacity = capacity
        self.gridParameters = gridParameters

    def heuristic(self,currentGrid,goalGrid):
        # Using Manhattan Distance as heuristic distance
        # currentGrid and goalGrid: [x,y,z]
        distX = abs(currentGrid[0] - goalGrid[0])
        distY = abs(currentGrid[1] - goalGrid[1])
        distZ = abs(currentGrid[2] - goalGrid[2])
        dist = distX + distY + distZ
        return dist

    def get_grid_neighbors(self,currentGrid,pinStart,pinEnd):

        # Key grid are grids containing pinStart or pinEnd
        keyGrid = [(pinStart[0], pinStart[1], pinStart[2]), (pinEnd[0], pinEnd[1], pinEnd[2])]

        # Store allowable neighbor positions in a list
        neighbor = []
        curX = currentGrid[0];curY = currentGrid[1];curZ = currentGrid[2]
        curLengthX = currentGrid[3]; curLengthY = currentGrid[4]

        possible_neighborGrid = [(curX-1,curY,curZ),(curX+1,curY,curZ),
                             (curX,curY-1,curZ),(curX,curY+1,curZ),
                             (curX, curY, curZ+1),(curX,curY,curZ-1)]
        for n in possible_neighborGrid:
            if n[0] < 0 or n[0] >= self.gridParameters['gridSize'][0] or\
                n[1] < 0 or n[1] >= self.gridParameters['gridSize'][1] or\
                 n[2] <= 0 or n[2] > self.gridParameters['gridSize'][2]:
                continue
            elif n == keyGrid[0]:
                n = pinStart
            elif n == keyGrid[1]:
                n = pinEnd
            else:
                neighborX = self.gridParameters['Origin'][0] + n[0]*self.gridParameters['tileWidth']
                neighborY = self.gridParameters['Origin'][1] + n[1]*self.gridParameters['tileHeight']
                n = n + (int(neighborX),int(neighborY))
            neighbor.append(n)

        # Output during search
        # print('Current Position', currentGrid)
        # print('Neighbor Position',neighbor)
        return neighbor

    def get_move_cost(self,state,action):
        if self.capacity[state[0],state[1],state[2]-1,action] <= 0:
            return 1000 # Very high cost for Overflow
        return 1 # Normal action cost

def AStarSearchRouter(pinStart, pinEnd, graph):
    # pin = (xGrid, yGrid, layer, xLength, yLength)

    # Key grid are grids containing pinStart or pinEnd
    keyGrid = [(pinStart[0],pinStart[1],pinStart[2]),(pinEnd[0],pinEnd[1],pinEnd[2])]

    G = {} # Actual Movement cost to each position from the start position
    F = {} # Estimated movement cost of start to end going via this position

    # Initializing starting value
    G[pinStart] = 0
    F[pinStart] = graph.heuristic(pinStart,pinEnd)

    closedVertices = set()
    openVertices = set([pinStart])
    cameFrom = {}

    # print('graph.capacity\n',graph.capacity[0,0,0,0])
    # Update capacity
    # graph.capacity = gridgraph.updateCapacityRL(graph.capacity,(0,0,1,5,5),0)


    while len(openVertices) > 0:
        # Get the vertex in the open list with lowest F score
        current = None
        currentFscore = None
        for pos in openVertices:
            if current is None or F[pos] < currentFscore:
                currentFscore = F[pos]
                current = pos
            # if current is None:
            #     current = pos
            # if currentFscore is None:
            #     currentFscore = graph.heuristic(current,pinEnd)
            
            # if  F[pos] < currentFscore:
            #     currentFscore = F[pos]
                
            #     action = 0
            #     delta = [pos[0]-current[0],pos[1]-current[1],pos[2]-current[2]]
            #     if delta[2] == 1:
            #         action = 4
            #     elif delta[2] == -1:
            #         action = 5
            #     elif delta[1] == 1:
            #         action = 2
            #     elif delta[1] == -1:
            #         action = 3
            #     elif delta[0] == 1:
            #         action = 0
            #     elif delta[0] == -1:
            #         action = 1
            #     oldcurrent = current
                
            #     current = pos
            #     # Update capacity

            #     graph.capacity = gridgraph.updateCapacityRL(graph.capacity,oldcurrent,action)
                       

        # Check if pinEnd is reached
        if current == pinEnd:
            # Retrace route backward
            path = [(current[3],current[4],current[2],current[0],current[1])]
            while current in cameFrom:
                current = cameFrom[current]
                path.append((current[3],current[4],current[2],current[0],current[1]))
            path.reverse()
            return path, F[pinEnd] # Routing accomplished

        # Mark the current vertex as closed
        openVertices.remove(current)
        closedVertices.add(current)

        # Update scores for vertices near the current position
        for neighbor in graph.get_grid_neighbors(current,pinStart,pinEnd):
            if neighbor in closedVertices:
                continue  #  We have already processed this node exhaustively
            # Get action based on current and neighbour
            delta = [neighbor[0]-current[0],neighbor[1]-current[1],neighbor[2]-current[2]]
            #  such as (0,0,1) means moving +z
            action = 0
            if delta == [0,0,1]:
                action = 4
            elif delta == [0,0,-1]:
                action = 5
            elif delta == [0,1,0]:
                action = 2
            elif delta == [0,-1,0]:
                action = 3
            elif delta == [1,0,0]:
                action = 0
            elif delta == [-1,0,0]:
                action = 1

            candidateG = G[current] + graph.get_move_cost(current,action)
            if neighbor not in openVertices:
                openVertices.add(neighbor) # Discovered a new vertex
            elif candidateG >= G[neighbor]:
                continue # This G score is worse than previously found


            # Adopt this G score
            cameFrom[neighbor] = current
            G[neighbor] = candidateG
            H = graph.heuristic(neighbor,pinEnd)
            F[neighbor] = G[neighbor] + H
        
        # update capacity
        # capacity = gridgraph.updateCapacityRL(capacity,state,action)

    raise RuntimeError("A* failed to find a solution")

if __name__ == "__main__":
    filename = 'small.gr'
    # filename = 'adaptec1.capo70.2d.35.50.90.gr'
    # filename = 'sampleBenchmark'

    # # Getting Net Info
    grid_info = init.read(filename)
    # print(grid_info)
    # # print(init.gridParameters(grid_info)['netInfo'])
    for item in init.gridParameters(grid_info).items():
        print(item)
    gridParameters = init.gridParameters(grid_info)
    # # for net in init.gridParameters(grid_info)['netInfo']:
    # init.GridGraph(init.gridParameters(grid_info)).show_grid()
    # init.GridGraph(init.gridParameters(grid_info)).pin_density_plot()

    # # GridGraph
    capacity = gridgraph.GridGraph(init.gridParameters(grid_info)).generate_capacity()
    # print(capacity[:,:,0,0])
    gridX,gridY,gridZ = gridgraph.GridGraph(init.gridParameters(grid_info)).generate_grid()
    # print(gridX[1,1,0])
    # print(gridY[1,1,0])
    # print(gridZ[1,1,0])

    # # Test: get grid neighbors
    # pinStart = (0,0,0,5,5); pinEnd = (2,0,0,25,5)
    # neighbor = AStarSearchGraph(gridParameters,capacity).get_grid_neighbors([1,1,0,15,15],pinStart,pinEnd)
    # print('Neighbor Test: ', neighbor)

    # Test: get move cost
    # state = (1,0,0); action = 1
    # cost = AStarSearchGraph(gridParameters,capacity).get_move_cost(state,action)
    # print(cost)


    # Test: A* router
    gridGraph = AStarSearchGraph(gridParameters,capacity)
    pinStart = (0,0,1,5,5); pinEnd = (2,0,1,25,5)
    route, cost = AStarSearchRouter(pinStart, pinEnd, gridGraph)
    print('Route',route)
    print('Cost',cost)


    # coord_L1 = []; coord_L0 = []
    # for coord in route:
    #     if coord[2] == 1:
    #         coord_L1.append(coord)
    #     else:
    #         coord_L0.append(coord)
    # plt.plot([coord[0] for coord in coord_L0],[coord[1] for coord in coord_L0],label='Layer 0',color='r')
    # plt.plot([coord[0] for coord in coord_L1],[coord[1] for coord in coord_L1],label='Layer 1',color='b')
    # plt.plot([coord[0] for coord in route],[coord[1] for coord in route],)

    # Possible bug: diagnol plot found in plots
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # ax.plot(route[0],route[1],route[2])
    x = [coord[0] for coord in route]
    y = [coord[1] for coord in route]
    z = [coord[2] for coord in route]
    ax.plot(x,y,z,'r',linewidth=2)

    # plt.xlim([0,324])
    # plt.ylim([0,324])
    # plt.legend()
    # plt.xlim([-1, gridParameters['gridSize'][0]])
    # plt.ylim([-1, gridParameters['gridSize'][1]])
    plt.show()
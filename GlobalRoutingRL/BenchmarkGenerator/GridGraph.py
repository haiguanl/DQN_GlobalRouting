import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import animation

import Initializer as init

# Create grid graph based on parsed input info

class GridGraph(object):
    def __init__(self,gridParameters):
        self.gridParameters = gridParameters
        return

    def generate_grid(self):
        # Initialize grid coordinates
        # Input: grid size
        gridX,gridY,gridZ = np.meshgrid(np.arange(self.gridParameters['gridSize'][0]), np.arange(self.gridParameters['gridSize'][1]),
                          np.arange(self.gridParameters['gridSize'][2]))
        # gridCoord = [gridX,gridY,gridZ]
        # print(gridX.shape)
        # gridCoord = np.zeros((3,3,2))
        # for i in range(self.gridParameters['gridSize'][0]):
        #     for j in range(self.gridParameters['gridSize'][1]):
        #         for k in range(self.gridParameters['gridSize'][2]):
        #             gridCoord[i,j,k] = [gridX[i,j,k],gridY[i,j,k],gridZ[i,j,k]]
        return gridX,gridY,gridZ

    def generate_capacity(self):
        # Input: VerticalCapacity, HorizontalCapacity, ReducedCapacity, MinWidth, MinSpacing
        # Update Input: Routed Nets Path
        # Capacity description direction:
        #[0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]

        capacity = np.zeros((self.gridParameters['gridSize'][0],self.gridParameters['gridSize'][1],
                         self.gridParameters['gridSize'][2],6))

        ## Apply initial condition to capacity
        # Calculate Available NumNet in each direction
        # Layer 0
        verticalNumNet = [self.gridParameters['verticalCapacity'][0]/
                          (self.gridParameters['minWidth'][0]+self.gridParameters['minSpacing'][0]),
                          self.gridParameters['verticalCapacity'][1] /
                          (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
        horizontalNumNet = [self.gridParameters['horizontalCapacity'][0]/
                          (self.gridParameters['minWidth'][0]+self.gridParameters['minSpacing'][0]),
                          self.gridParameters['horizontalCapacity'][1] /
                          (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
        # print(horizontalNumNet)
        # Apply available NumNet to grid capacity variables
        capacity[:,:,0,0] = capacity[:,:,0,1] = horizontalNumNet[0]
        capacity[:,:,1,0] = capacity[:,:,1,1] = horizontalNumNet[1]
        capacity[:,:,0,2] = capacity[:,:,0,3] = verticalNumNet[0]
        capacity[:,:,1,2] = capacity[:,:,1,3] = verticalNumNet[1]

        # Assume Via Ability to be very large
        capacity[:,:,0,4] = 10; capacity[:,:,1,5] = 10

        # Apply Reduced Ability
        for i in range(int(self.gridParameters['reducedCapacity'][0])):
            # print('Apply reduced capacity operation')
            delta = [self.gridParameters['reducedCapacitySpecify'][str(i+1)][0]-
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1] -
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2] -
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5],
                     ]
            if delta[0] != 0:
                # print(self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0]-1)
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int((delta[0]+1)/2)] = \
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int((-delta[0]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
            elif delta[1] != 0:
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int(2+(delta[1]+1)/2)] = \
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int(2+(-delta[1]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
            elif delta[2] != 0:
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int(4+(delta[2]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int(4+(-delta[2]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]

        # Remove edge capacity
        capacity[:,:,1,4] = 0; capacity[:,:,0,5] = 0 # Z-direction edge capacity edge removal
        capacity[:,0,:,3] = 0; capacity[:,self.gridParameters['gridSize'][1]-1,:,2] = 0 # Y-direction edge capacity edge removal
        capacity[0,:,:,1] = 0; capacity[self.gridParameters['gridSize'][0]-1,:,:,0] = 0 # X-direction edge capacity edge removal
        return capacity

    # def generate_capacity(self):
    #     # Input: VerticalCapacity, HorizontalCapacity, ReducedCapacity, MinWidth, MinSpacing
    #     # Update Input: Routed Nets Path
    #     # Capacity description direction:
    #     #[0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]
    #     capacity = np.zeros((self.gridParameters['gridSize'][0],self.gridParameters['gridSize'][1],
    #                      self.gridParameters['gridSize'][2],6))

    #     ## Apply initial condition to capacity
    #     # Calculate Available NumNet in each direction
    #     # Layer 0
    #     verticalNumNet = [self.gridParameters['verticalCapacity'][0]/
    #                       (self.gridParameters['minWidth'][0]+self.gridParameters['minSpacing'][0]),
    #                       self.gridParameters['verticalCapacity'][1] /
    #                       (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
    #     horizontalNumNet = [self.gridParameters['horizontalCapacity'][0]/
    #                       (self.gridParameters['minWidth'][0]+self.gridParameters['minSpacing'][0]),
    #                       self.gridParameters['horizontalCapacity'][1] /
    #                       (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
    #     # print(horizontalNumNet)
    #     # Apply available NumNet to grid capacity variables
    #     capacity[:,:,0,0] = capacity[:,:,0,1] = horizontalNumNet[0]
    #     capacity[:,:,1,0] = capacity[:,:,1,1] = horizontalNumNet[1]
    #     capacity[:,:,0,2] = capacity[:,:,0,3] = verticalNumNet[0]
    #     capacity[:,:,1,2] = capacity[:,:,1,3] = verticalNumNet[1]



    #     # Assume Via Ability to be ???
    #     capacity[:,:,0,4] = 10; capacity[:,:,1,5] = 10
    #     # Apply Reduced Ability
    #     for i in range(int(self.gridParameters['reducedCapacity'][0])):
    #         delta = [self.gridParameters['reducedCapacitySpecify'][str(i+1)][0]-
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1] -
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2] -
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5],
    #                  ]
    #         # print(self.gridParameters['reducedCapacitySpecify'][str(i+1)])
    #         # print(self.gridParameters['reducedCapacitySpecify'][str(i+1)][0])

    #         if delta[0] != 0:
    #             # print('X_Capacity_Change')
    #             capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int((delta[0]+1)/2)] = \
    #             capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int((-delta[0]+1)/2)] = \
    #                 self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
    #         elif delta[1] != 0:
    #             # print('Y_Capacity_Change')
    #             capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int(2+(delta[1]+1)/2)] = \
    #             capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int(2+(-delta[1]+1)/2)] = \
    #                 self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
    #         elif delta[2] != 0:
    #             # print('Z_Capacity_Change')
    #             capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int(4+(delta[2]+1)/2)] = \
    #                 self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
    #             capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
    #                  self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int(4+(-delta[2]+1)/2)] = \
    #                 self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]

    #     # Remove edge capacity
    #     capacity[:,:,1,4] = 0; capacity[:,:,0,5] = 0 # Z-direction edge capacity edge removal
    #     capacity[:,0,:,3] = 0; capacity[:,self.gridParameters['gridSize'][1]-1,:,2] = 0 # Y-direction edge capacity edge removal
    #     capacity[0,:,:,1] = 0; capacity[self.gridParameters['gridSize'][0]-1,:,:,0] = 0 # X-direction edge capacity edge removal
    #     return capacity

    # def pin_density(self):
    #     # Input: pin location globally
    #     return


    def step(self,state,action):
        nextState = ()
        if action == 0:
            nextState = (state[0]+1,state[1],state[2])
        elif action == 1:
            nextState = (state[0]-1, state[1], state[2])
        elif action == 2:
            nextState = (state[0], state[1]+1, state[2])
        elif action == 3:
            nextState = (state[0], state[1]-1, state[2])
        elif action == 4:
            nextState = (state[0], state[1], state[2]+1)
        elif action == 5:
            nextState = (state[0], state[1], state[2]-1)
        print('Output')
        return nextState

    def reward(self,state,action):
        return

def updateCapacity(capacity,route):
    for i in range(len(route)-1):
        diff = [route[i+1][3]-route[i][3],
                route[i+1][4]-route[i][4],
                route[i+1][2]-route[i][2]]
        if diff[0] == 1:
            capacity[route[i][3],route[i][4],route[i][2]-1,0] -= 1
            capacity[route[i+1][3], route[i+1][4], route[i+1][2]-1,1] -= 1
        elif diff[0] == -1:
            capacity[route[i][3], route[i][4], route[i][2]-1,1] -= 1
            capacity[route[i+1][3], route[i+1][4], route[i+1][2]-1, 0] -= 1
        elif diff[1] == 1:
            capacity[route[i][3],route[i][4],route[i][2]-1,2] -= 1
            capacity[route[i+1][3], route[i+1][4], route[i+1][2]-1,3] -= 1
        elif diff[1] == -1:
            capacity[route[i][3],route[i][4],route[i][2]-1,3] -= 1
            capacity[route[i+1][3], route[i+1][4], route[i+1][2]-1,2] -= 1
        elif diff[2] == 1:
            capacity[route[i][3],route[i][4],route[i][2]-1,4] -= 1
            capacity[route[i+1][3], route[i+1][4], route[i+1][2]-1,5] -= 1
        elif diff[2] == -1:
            capacity[route[i][3],route[i][4],route[i][2]-1,5] -= 1
            capacity[route[i+1][3], route[i+1][4], route[i+1][2]-1,4] -= 1
    return capacity

def updateCapacityRL(capacity,state,action):
    # capacity could go to negative

    # capacity[xgrid,ygrid,z={0,1},0~5]
    # state [xgrid,ygrid,zlayer={1,2},xlength,ylength]
    # action
    if action == 0:
        capacity[state[0],state[1],state[2]-1,0] -= 1
        capacity[state[0]+1,state[1],state[2]-1,1] -= 1
    elif action == 1:
        capacity[state[0],state[1],state[2]-1,1] -= 1
        capacity[state[0]-1,state[1],state[2]-1,0] -= 1
    elif action == 2:
        capacity[state[0],state[1],state[2]-1,2] -= 1
        capacity[state[0],state[1]+1,state[2]-1,3] -= 1
    elif action == 3:
        capacity[state[0],state[1],state[2]-1,3] -= 1
        capacity[state[0],state[1]-1,state[2]-1,2] -= 1
    elif action == 4:
        capacity[state[0],state[1],state[2]-1,4] -= 1
        capacity[state[0],state[1],state[2],5] -= 1
    elif action == 5:
        capacity[state[0],state[1],state[2]-1,5] -= 1
        capacity[state[0],state[1],state[2]-2,4] -= 1
    return capacity

if __name__ == '__main__':
    filename = 'small.gr'
    # filename = 'adaptec1.capo70.2d.35.50.90.gr'
    # filename = 'sampleBenchmark'

    # Getting Net Info
    grid_info = init.read(filename)
    print(grid_info)

    # # # print(init.gridParameters(grid_info)['netInfo'])
    #
    for item in init.gridParameters(grid_info).items():
        print(item)

    # # for net in init.gridParameters(grid_info)['netInfo']:
    # #     print (net)
    # init.GridGraph(init.gridParameters(grid_info)).show_grid()
    # init.GridGraph(init.gridParameters(grid_info)).pin_density_plot()
    #
    # capacity = GridGraph(init.gridParameters(grid_info)).generate_capacity()
    # print(capacity[:,:,0,1])
    # gridX, gridY, gridZ= GridGraph(init.gridParameters(grid_info)).generate_grid()
    # print(gridX[1,1,0])
    # print(gridY[1,1,0])
    # print(gridZ[1,1,0])


    # # Check capacity update
    # print(capacity[0, 1, 0, 4])
    # RouteListMerged = [[(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0), (1, 2, 0),
    #   (2, 2, 0), (2, 2, 1), (2, 1, 1), (2, 0, 1), (2, 0, 0)]]
    # capacity = updateCapacity(capacity,RouteListMerged)
    # print(capacity[0, 1, 0, 4])



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import sklearn.neighbors as neighbor
import KDTree as kd

from matplotlib import cm
from matplotlib import animation


class GridGraph(object):
    def __init__(self,gridParameters):
        self.gridParameters = gridParameters
        return
    def show_grid(self):
        fig, axes = plt.subplots(1,self.gridParameters['gridSize'][2])
        tile_width = self.gridParameters['tileWidth']
        tile_height = self.gridParameters['tileHeight']
        for i in range(self.gridParameters['gridSize'][2]):
            axes[i].imshow(np.random.random((self.gridParameters['gridSize'][0],self.gridParameters['gridSize'][1]))
                           ,origin='lower',extent=(0,tile_width*self.gridParameters['gridSize'][0],0,tile_height*self.gridParameters['gridSize'][1]),
                           alpha=0.7,cmap=cm.gray)
            axes[i].set(title='Layer %i' %i)
            # Visualize capacity
            if self.gridParameters['verticalCapacity'][i] == 0:
                for k in range(self.gridParameters['gridSize'][1]-1):
                    for j in range(self.gridParameters['gridSize'][0]):
                        rect = patches.Rectangle((0.2*tile_width+tile_width*j,0.9*tile_height+tile_height*k),0.6*tile_width,0.2*tile_height,facecolor='b')
                        axes[i].add_patch(rect)
            if self.gridParameters['horizontalCapacity'][i] == 0:
                for k in range(self.gridParameters['gridSize'][1] ):
                    for j in range(self.gridParameters['gridSize'][0]-1):
                        rect = patches.Rectangle((0.9*tile_width +tile_width*j,0.2*tile_height + tile_height*k),0.2*tile_width,0.6*tile_height, facecolor='b')
                        axes[i].add_patch(rect)

        for i in range(4,len(gridParameters(grid_info)['netInfo'][0])):
            pinCoord = gridParameters(grid_info)['netInfo'][0][str(i-3)]
            axes[pinCoord[2]].plot(pinCoord[0],pinCoord[1],'x')
        plt.show()
        return
    def pin_density_plot(self):
        pin_XList = []; pin_YList = []
        for i in self.gridParameters['netInfo']:
            for j in range(i['numPins']):
                pin_XList.append(i[str(j + 1)][0])
                pin_YList.append(i[str(j + 1)][1])
        print('pinX',pin_XList)
        plt.xlim([0, self.gridParameters['tileWidth'] * self.gridParameters['gridSize'][0]])
        plt.ylim([0, self.gridParameters['tileHeight'] * self.gridParameters['gridSize'][1]])
        plt.plot(pin_XList,pin_YList,'b.')
        plt.xlabel('X Lengths')
        plt.ylabel('Y Lengths')
        plt.show()

    def kdTree(self):
        pin_XList = []
        pin_YList = []
        for i in self.gridParameters['netInfo']:
            for j in range(i['numPins']):
                pin_XList.append(i[str(j + 1)][0])
                pin_YList.append(i[str(j + 1)][1])
        pinLoc = np.column_stack((pin_XList,pin_YList))
        # gridNum = 2
        # tree = neighbor.KDTree(pinLoc, leaf_size=int(gridNum))
        #
        # print(tree.idx_array)

        KDT = kd.KDTree(pinLoc, [np.min(pinLoc[:,0]),np.min(pinLoc[:,1])], [np.max(pinLoc[:,0]),np.max(pinLoc[:,1])])
        # ------------------------------------------------------------
        # Plot four different levels of the KD tree
        fig = plt.figure(figsize=(5, 5))
        fig.subplots_adjust(wspace=0.1, hspace=0.15,
                            left=0.1, right=0.9,
                            bottom=0.05, top=0.9)

        for level in range(1, 10):
            ax = fig.add_subplot(3, 3, level, xticks=[], yticks=[])
            ax.scatter(pinLoc[:, 0], pinLoc[:, 1], s=9)
            KDT.draw_rectangle(ax, depth=level )
            #
            # ax.set_xlim(-1.2, 1.2)
            # ax.set_ylim(-0.15, 1.15)
            ax.set_title('level %i' % level)

        # suptitle() adds a title to the entire figure
        fig.suptitle('kd-tree Example')
        plt.show()
        #
        # plt.subplot(2,2,1);plt.title('Pin No.8000-9000')
        # plt.plot(pin_XList[8000:9000],pin_YList[8000:9000],'b.')
        # plt.subplot(2,2,2);plt.title('Pin No.20000-21000')
        # plt.plot(pin_XList[20000:21000],pin_YList[20000:21000],'b.')
        # plt.subplot(2,2,3);plt.title('Pin No.50000-51000')
        # plt.plot(pin_XList[50000:51000],pin_YList[50000:51000],'b.')
        # plt.subplot(2,2,4);plt.title('Pin No.100000-101000')
        # plt.plot(pin_XList[100000:101000],pin_YList[100000:101000],'b.')
        # plt.show()
        return

class Router(object):
    def __init__(self):
        return

def read(grfile):
    file = open(grfile,'r')
    grid_info = {}
    i = 0
    for line in file:
        if not line.strip():
            continue
        else:
            grid_info[i]= line.split()
        i += 1
    file.close()
    return grid_info

# Parsing input data
def gridParameters(grid_info):
    gridParameters = {}
    gridParameters['gridSize'] = [int(grid_info[0][1]),int(grid_info[0][2]),int(grid_info[0][3])]
    gridParameters['verticalCapacity'] = [float(grid_info[1][2]),float(grid_info[1][3])]
    gridParameters['horizontalCapacity'] = [float(grid_info[2][2]), float(grid_info[2][3])]
    gridParameters['minWidth'] = [float(grid_info[3][2]), float(grid_info[3][3])]
    gridParameters['minSpacing'] = [float(grid_info[4][2]), float(grid_info[4][3])]
    gridParameters['viaSpacing'] = [float(grid_info[5][2]), float(grid_info[5][3])]
    gridParameters['Origin'] = [float(grid_info[6][0]), float(grid_info[6][1])]
    gridParameters['tileWidth'] = float(grid_info[6][2]); gridParameters['tileHeight'] = float(grid_info[6][3])
    gridParameters['reducedCapacitySpecify'] = {}
    for lineNum in range(len(grid_info)):
        if 'num' in grid_info[lineNum]:
            gridParameters['numNet'] = int(grid_info[lineNum][2])
    netNum = 0
    pinEnumerator = 1; lineEnumerator = 8
    netParametersStore = []
    for lineNum in range(7,len(grid_info)):
        if 'A' in grid_info[lineNum][0]:
            netParameters = {}
            netParameters['netName'] = grid_info[lineNum][0]
            netParameters['netID'] = int(grid_info[lineNum][1])
            netParameters['numPins'] = int(grid_info[lineNum][2])
            netParameters['minWidth'] = float(grid_info[lineNum][3])
            pinNum = 1
            while ('A' not in grid_info[lineNum+pinNum][0]) and (len(grid_info[lineNum+pinNum])>1):
                netParameters[str(pinNum)] = [int(grid_info[lineNum+pinNum][0]),int(grid_info[lineNum+pinNum][1]),
                                             int(grid_info[lineNum+pinNum][2])]
                pinNum += 1

            gridParameters['reducedCapacity'] = grid_info[lineNum+pinNum]
            pinEnumerator = pinNum
            lineEnumerator = lineNum + pinNum + 1
            netParametersStore.append(netParameters)
        if ('n' in grid_info[lineNum][0])and (grid_info[lineNum][0] != 'num'):
            netParameters = {}
            netParameters['netName'] = grid_info[lineNum][0]
            netParameters['netID'] = int(grid_info[lineNum][1])
            netParameters['numPins'] = int(grid_info[lineNum][2])
            netParameters['minWidth'] = float(grid_info[lineNum][3])
            pinNum = 1
            while ('n' not in grid_info[lineNum+pinNum][0]) and (len(grid_info[lineNum+pinNum])>1):
                netParameters[str(pinNum)] = [int(grid_info[lineNum+pinNum][0]),int(grid_info[lineNum+pinNum][1]),
                                             int(grid_info[lineNum+pinNum][2])]
                pinNum += 1

            gridParameters['reducedCapacity'] = grid_info[lineNum+pinNum]
            pinEnumerator = pinNum
            lineEnumerator = lineNum + pinNum + 1
            netParametersStore.append(netParameters)
        gridParameters['netInfo'] = netParametersStore
    # Parsing adjustments depicting reduced capacity (override layer specification)
    i = 1
    for lineNum in range(lineEnumerator, len(grid_info)):
        gridParameters['reducedCapacitySpecify'][str(i)] = grid_info[lineNum]
        i += 1

    return gridParameters




if __name__ == '__main__':
    # filename = 'small.gr'
    filename = 'adaptec1.capo70.2d.35.50.90.gr'
    # filename = 'sampleBenchmark'

    grid_info = read(filename)
    print(grid_info)
    # print(gridParameters(grid_info)['netInfo'])

    for item in gridParameters(grid_info).items():
        print(item)
    #
    # for net in gridParameters(grid_info)['netInfo']:
    #     print (net)

    # GridGraph(gridParameters(grid_info)).show_grid()
    GridGraph(gridParameters(grid_info)).pin_density_plot()
    GridGraph(gridParameters(grid_info)).kdTree()
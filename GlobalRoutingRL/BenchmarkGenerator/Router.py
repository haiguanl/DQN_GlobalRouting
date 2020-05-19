from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import Initializer as init
import GridGraph as graph
import TwoPinRouterASearch as twoPinASearch
import MST as tree

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import operator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def getGridCoord(pinInGrid):
    # pin = (gridx,gridz,layer,actualx,actualy)
    return


if __name__ == "__main__":
    # filename = '3d.txt'
#    filename = '8by8small.gr'
#    filename = '8by8simplee.gr'
#     filename = '32by32simple.gr'
    # filename = '4by4simple.gr'
    # filename = '4by4simple_val2.gr'  # 3-by-3 nets toy sample
    # filename = 'adaptec1.capo70.2d.35.50.90.gr'
    filename = 'test_benchmark.gr'

    # # Getting Net Info
    grid_info = init.read(filename)
    # print(grid_info)
    # # print(init.gridParameters(grid_info)['netInfo'])
    # for item in init.gridParameters(grid_info).items():
        # print(item)
    gridParameters = init.gridParameters(grid_info)
    # # for net in init.gridParameters(grid_info)['netInfo']:
    # init.GridGraph(init.gridParameters(grid_info)).show_grid()
    # init.GridGraph(init.gridParameters(grid_info)).pin_density_plot()

    # # GridGraph
    capacity = graph.GridGraph(init.gridParameters(grid_info)).generate_capacity()
    # print(capacity[:,:,0,0])

    gridX,gridY,gridZ = graph.GridGraph(init.gridParameters(grid_info)).generate_grid()
    # print(gridX[1,1,0])
    # print(gridY[1,1,0])
    # print(gridZ[1,1,0])

    # Real Router for Multiple Net
    # Note: pinCoord input as absolute length coordinates
    gridGraph = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

    # Sort net
    halfWireLength = init.VisualGraph(init.gridParameters(grid_info)).bounding_length()
    # print('Half Wire Length:',halfWireLength)

    sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=True) # Large2Small
    # sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=False) # Small2Large

    # print('Sorted Half Wire Length:',sortedHalfWireLength)

    # Following plot is foe half wire length distribution
    # plt.figure()
    # plt.hist([i[1] for i in sortedHalfWireLength],bins=20)
    # plt.show()

    routeListMerged = []
    routeListNotMerged = []

    # For testing first part nets

    # for i in range(1):
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):

        # Determine nets to wire based on sorted nets (stored in list sortedHalfWireLength)
        # print('*********************')
        # print('Routing net No.',init.gridParameters(grid_info)['netInfo'][int(sortedHalfWireLength[i][0])]['netName'])
        # (above output is to get actual netName)
        # print('Routing net No.',sortedHalfWireLength[i][0])

        netNum = int(sortedHalfWireLength[i][0])

        # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree

        # # Remove pins that are in the same grid:
        netPinList = []
        netPinCoord = []
        for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
                             int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
                             int(gridParameters['netInfo'][netNum][str(j+1)][2]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][0]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])

        twoPinList = []
        for i in range(len(netPinList)-1):
            pinStart = netPinList[i]
            pinEnd = netPinList[i+1]
            twoPinList.append([pinStart,pinEnd])
            # Sort
            #
            # for i in range(len(netPinList)):
            #     for j in range(i+1,len(netPinList)):
            #         redundancy = (0,0,0,0,0)
            #         if netPinList[i][:3] == netPinList[j][:3]:
            #             redundancy = netPinList[i]
            #         netPinList.remove(redundancy)



        # for j in range(1,gridParameters['netInfo'][netNum]['numPins']):
        #     # print(gridParameters['netInfo'][i][str(j)][0])
        #
        #     # Here, pin representation is converted from length coordinates to grid coordinates,
        #     # while length coordinates is stored at last two positons
        #     pinStart = tuple([int((gridParameters['netInfo'][netNum][str(j)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
        #                      int((gridParameters['netInfo'][netNum][str(j)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
        #                      int(gridParameters['netInfo'][netNum][str(j)][2]),
        #                       int(gridParameters['netInfo'][netNum][str(j)][0]),
        #                       int(gridParameters['netInfo'][netNum][str(j)][1])])
        #     pinEnd = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
        #                      int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
        #                      int(gridParameters['netInfo'][netNum][str(j+1)][2]),
        #                     int(gridParameters['netInfo'][netNum][str(j+1)][0]),
        #                     int(gridParameters['netInfo'][netNum][str(j+1)][1])])
        #
        #     # Remove pin pairs that are in the same grid
        #     if pinStart[:3] == pinEnd[:3]:
        #         continue
        #     else:
        #         twoPinList.append([pinStart,pinEnd])

        # print('Two pin list:',twoPinList,'\n')

        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)
        # print('Two pin list after:', twoPinList, '\n')

        # Remove pin pairs that are in the same grid again
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])

        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        i = 1
        routeListSingleNet = []
        for twoPinPair in twoPinList:
            pinStart = twoPinPair[0]; pinEnd =  twoPinPair[1]
            # print('Routing pin pair No.',i)
            # print('Pin start ',pinStart)
            route, cost = twoPinASearch.AStarSearchRouter(pinStart, pinEnd, gridGraph)
            # print('Route:',route)
            # print('Cost:',cost)
            routeListSingleNet.append(route)
            i += 1


        # print('Route List Single Net:',routeListSingleNet,'\n')
        mergedrouteListSingleNet = []

        for list in routeListSingleNet:
            # if len(routeListSingleNet[0]) == 2:
            #     mergedrouteListSingleNet.append(list[0])
            #     mergedrouteListSingleNet.append(list[1])
            # else:
            for loc in list:
                    # if loc not in mergedrouteListSingleNet:
                    mergedrouteListSingleNet.append(loc)
        # print('Merged Route List Single Net',mergedrouteListSingleNet,'\n')
        routeListMerged.append(mergedrouteListSingleNet)
        routeListNotMerged.append(routeListSingleNet)

        # Update capacity and grid graph after routing one pin pair
        # # WARNING: capacity update could lead to failure since capacity could go to negative (possible bug)
        # # # print(route)
        capacity = graph.updateCapacity(capacity, mergedrouteListSingleNet)
        # gridGraph = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

    # Plot of routing for multilple net
    # print('\nRoute List Merged:',routeListMerged)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0.5,2.0)
    #
    for routeList in routeListNotMerged:
        for route in routeList:
            x = [coord[3] for coord in route]
            y = [coord[4] for coord in route]
            z = [coord[2] for coord in route]
            ax.plot(x,y,z)


    plt.xlim([0, gridParameters['gridSize'][0]-1])
    plt.ylim([0, gridParameters['gridSize'][1]-1])
    # plt.
    plt.savefig('RoutingVisualize.jpg')
    fig.tight_layout()
    plt.show()

    # for i in range(len(routeListMerged)):
    #     print(i)
    #     print(routeListMerged[i])

    #Generate output file
    # print('routeListMerged',routeListMerged)
    f = open('%s.solutiontesting' % filename, 'w+')

    # For testing first part nets

    # for i in range(1):
    for i in range(gridParameters['numNet']):
        indicator = i
        netNum = int(sortedHalfWireLength[i][0])
        i = netNum

        value = '{netName} {netID} {cost}\n'.format(netName=gridParameters['netInfo'][i]['netName'],
                                              netID = gridParameters['netInfo'][i]['netID'],
                                              cost = max(0,len(routeListMerged[indicator])-1))
        f.write(value)
        for j in range(len(routeListMerged[indicator])-1):
        # In generating the route in length coordinate system, the first pin (corresponding to griParameters['netInfo'][i]['1'])
        # is used as reference point
            a = routeListMerged[indicator][j]
            b = routeListMerged[indicator][j+1]
            diff = [abs(a[2]-b[2]),abs(a[3]-b[3]),abs(a[4]-b[4])]
            if diff[1] > 2 or diff[2] > 2:
                continue
            elif diff[1] == 2 or diff[2] == 2:
                # print('Alert')
                continue
            elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                continue
            elif diff[0] + diff[1] + diff[2] >= 2:
                continue
            else:
                value = '({},{},{})-({},{},{})\n'.format(a[0],a[1],a[2],b[0],b[1],b[2])
                f.write(value)
        f.write('!\n')
    f.close()

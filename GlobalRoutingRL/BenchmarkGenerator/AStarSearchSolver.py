#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:07:32 2019

@author: liaohaiguang
"""

import os
import operator
import numpy as np
import matplotlib.pyplot as plt

import Initializer as init
import GridGraph as graph
import TwoPinRouterASearch as twoPinASearch
import MST as tree



# The following function solve benchmarks with A* Search Algorithm 
# and return edge traffic information
def solve(benchmarkfile,solution_savepath):
    # Getting Net Info
    grid_info = init.read(benchmarkfile)
    gridParameters = init.gridParameters(grid_info)
    capacity = graph.GridGraph(init.gridParameters(grid_info)).generate_capacity()
    gridX,gridY,gridZ = graph.GridGraph(init.gridParameters(grid_info)).generate_grid()
    gridGraph = twoPinASearch.AStarSearchGraph(gridParameters, capacity)
    # Sort net
    halfWireLength = init.VisualGraph(init.gridParameters(grid_info)).bounding_length()
    sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=True) # Large2Small
    # print('sortedHalfWireLength',sortedHalfWireLength)
    
    netSort = []
    for i in range(len(sortedHalfWireLength)):
        netSort.append(int(sortedHalfWireLength[i][0]))
    # print('netSort',netSort)

    routeListMerged = []
    routeListNotMerged = []
    routeListMergedCap = []
    twoPinListPlot = []
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        # print('*********************')
        # print('Routing net No.',sortedHalfWireLength[i][0])
        # netNum = int(sortedHalfWireLength[i][0])
        netNum = netSort[i]
        # print('netNum',netNum)
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
        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)
        # Remove pin pairs that are in the same grid again
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])

        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])
        i = 1
        twoPinListPlot.append(twoPinList)
        routeListSingleNet = []
        routeListSingleNetPlot = []
        for twoPinPair in twoPinList:
            pinStart = twoPinPair[0]; pinEnd =  twoPinPair[1]
            route, cost = twoPinASearch.AStarSearchRouter(pinStart, pinEnd, gridGraph)
            routeListSingleNet.append([route])
            routeListSingleNetPlot.append(route)
            i += 1
        # print('Route List Single Net:',routeListSingleNet)
        mergedrouteListSingleNet = []
        mergedrouteListSingleNetCap = []

        for twoPinRoute in routeListSingleNet:
            for twoPinRouteSpecific in twoPinRoute:
                    # if loc not in mergedrouteListSingleNet:
                    mergedrouteListSingleNet.append(twoPinRouteSpecific)
                    for i in range(len(twoPinRouteSpecific)):
                        mergedrouteListSingleNetCap.append(twoPinRouteSpecific[i])
        routeListMerged.append(mergedrouteListSingleNet)
        routeListMergedCap.append(mergedrouteListSingleNetCap)

        routeListNotMerged.append(routeListSingleNetPlot)
        # Update capacity and grid graph after routing one pin pair
        # # WARNING: capacity update could lead to failure since capacity could go to negative (possible bug)
        # # # print(route)
        capacity = graph.updateCapacity(capacity, mergedrouteListSingleNetCap)
        # gridGraph = twoPinASearch.AStarSearchGraph(gridParameters, capacity)
    # print('\nRoute List Merged:',routeListMerged)
    twoPinListPlotRavel = []
    for i in range(len(twoPinListPlot)):
        for j in range(len(twoPinListPlot[i])):
            twoPinListPlotRavel.append(twoPinListPlot[i][j][0])
            twoPinListPlotRavel.append(twoPinListPlot[i][j][1])
    
    # Visualize results on 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # # Hide grid lines
    # ax.grid(False)
    # # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # plt.axis('off')
    # plt.grid(b=None)
    ax.set_zlim(0.75,2.25)
    plt.axis('off')
    #
    x_meshP = np.linspace(0,gridParameters['gridSize'][0]-1,200)
    y_meshP = np.linspace(0,gridParameters['gridSize'][1]-1,200)
    z_meshP = np.linspace(1,2,200)
    x_mesh,y_mesh = np.meshgrid(x_meshP,y_meshP)
    z_mesh = np.ones_like(x_mesh)
    ax.plot_surface(x_mesh,y_mesh,z_mesh,alpha=0.3,color='r')
    ax.plot_surface(x_mesh,y_mesh,2*z_mesh,alpha=0.3,color='r')




    # for i in twoPinListPlotRavel:
        # x = [coord[0] for coord in twoPinListPlotRavel]
        # y = [coord[1] for coord in twoPinListPlotRavel]
        # z = [coord[2] for coord in twoPinListPlotRavel]
        # ax.scatter(x,y,z,s=25,facecolors='none', edgecolors='k')

    for routeList in routeListNotMerged:
        for route in routeList:
            x = [coord[3] for coord in route]
            y = [coord[4] for coord in route]
            z = [coord[2] for coord in route]
            ax.plot(x,y,z,linewidth=2.5)

    plt.xlim([0, gridParameters['gridSize'][0]-1])
    plt.ylim([0, gridParameters['gridSize'][1]-1])
    plt.savefig('{path}RoutingVisualize_{benchmark}.jpg'.format(path=solution_savepath,\
                benchmark=benchmarkfile))
#    plt.show()
    plt.close()

    # Visualize results on 2D 
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_zlim(-0.5,2.5)
    ax = fig.add_subplot(111)
    #
    for routeList in routeListNotMerged:
        for route in routeList:
            num_points = len(route)
            for i in range(num_points-1):
                pair_x = [route[i][3], route[i+1][3]]
                pair_y = [route[i][4], route[i+1][4]]
                pair_z = [route[i][2], route[i+1][2]]
                if pair_z[0] ==pair_z[1] == 1:
                    ax.plot(pair_x, pair_y, color='blue', linewidth=2.5)
                if pair_z[0] ==pair_z[1] == 2:
                    ax.plot(pair_x, pair_y, color='red', linewidth=2.5)
    ax.axis('scaled')
    ax.invert_yaxis()
    plt.xlim([-0.1, gridParameters['gridSize'][0]-0.9])
    plt.ylim([-0.1, gridParameters['gridSize'][1]-0.9])
    plt.axis('off')
    # for i in twoPinListPlotRavel:
    #     x = [coord[0] for coord in twoPinListPlotRavel]
    #     y = [coord[1] for coord in twoPinListPlotRavel]
    #     ax.scatter(x,y,s=40, facecolors='none', edgecolors='k')
    plt.savefig('{path}RoutingVisualize_2D_{benchmark}.jpg'.format(path=solution_savepath,\
                benchmark=benchmarkfile))
    plt.close()

    # dump solution for Astar
    f = open('{solution_savepath}{benchmark}Astar_solution'.format(\
                solution_savepath=solution_savepath,benchmark=benchmarkfile),'w+')
    
    for i in range(gridParameters['numNet']):
        singleNetRouteCache = []
        indicator = i
        # netNum = int(sortedHalfWireLength[i][0])
        netNum = netSort[i]
        i = netNum

        value = '{netName} {netID} {cost}\n'.format(netName=gridParameters['netInfo'][i]['netName'],
                                              netID = gridParameters['netInfo'][i]['netID'],
                                              cost = max(0,len(routeListMerged[indicator])-1))
        f.write(value)
        for j in range(len(routeListMerged[indicator])):
        # In generating the route in length coordinate system, the first pin (corresponding to griParameters['netInfo'][i]['1'])
        # is used as reference point
            for k in range(len(routeListMerged[indicator][j])-1):
                a = routeListMerged[indicator][j][k]
                b = routeListMerged[indicator][j][k+1]

                if (a[3],a[4],a[2],b[3],b[4],b[2]) not in singleNetRouteCache:
                    singleNetRouteCache.append((a[3],a[4],a[2],b[3],b[4],b[2]))
                    singleNetRouteCache.append((b[3],b[4],b[2],a[3],a[4],a[2]))

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
    return routeListMergedCap

if __name__ == "__main__":
    
    # get edge traffic (data structure: gridSize*gridSize*2)
#    edge_traffic =
    os.chdir("benchmark/")
    benchmarkfile = "test_benchmark_5.gr"
    
    
    solution_savepath = '../solutionsForBenchmark/' 
    os.mkdir(solution_savepath)

    solve(benchmarkfile,solution_savepath)
    
    
    
    
    
    
    
    
    
    

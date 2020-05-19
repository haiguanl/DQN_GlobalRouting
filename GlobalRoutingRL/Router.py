from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import Initializer as init
import GridGraph as graph
import TwoPinRouterASearch as twoPinASearch
import MST as tree
import keras, tensorflow as tf, numpy as np, sys, copy, argparse, random
import operator
import math
import os
import DQN_Implementation
import pickle as pkl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import random

np.random.seed(10701)
tf.set_random_seed(10701)
random.seed(10701)

def DRL_implementation(filename,globali):
    try:
        # Filename corresponds to benchmark to route
        # filename = '3d.txt'
        # filename = '8by8small.gr'  # 8-by-8 10 nets toy sample
        # filename = '8by8simplee.gr'  # 8-by-8 10 nets toy sample
        # filename = 'adaptecSmall.gr'
        # filename = '4by4small.gr'  # 3-by-3 nets toy sample
    #    filename = '4by4simple.gr'  # 3-by-3 nets toy sample
        # filename = 'adaptec1.capo70.2d.35.50.90.gr'

        # # Getting Net Info
        grid_info = init.read(filename)
        gridParameters = init.gridParameters(grid_info)

        # # GridGraph
        graphcase = graph.GridGraph(init.gridParameters(grid_info))
        capacity = graphcase.generate_capacity()
        # print ('capacity before route: ',capacity.shape)

        gridX,gridY,gridZ = graphcase.generate_grid()

        # Real Router for Multiple Net
        # Note: pinCoord input as absolute length coordinates
        gridGraphSearch = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

        # Sort net
        halfWireLength = init.VisualGraph(init.gridParameters(grid_info)).bounding_length()
    #    print('Half Wire Length:',halfWireLength)

        sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=True) # Large2Small
        # sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=False) # Small2Large

        netSort = []
        for i in range(gridParameters['numNet']):
            order = int(sortedHalfWireLength[i][0])
            netSort.append(order)
        # random order the nets
        # print('netSort Before',netSort)
        # random.shuffle(netSort)
        # print('netSort After',netSort)

        routeListMerged = []
        routeListNotMerged = []

        print('gridParameters',gridParameters)
        # Getting two pin list combo (For RL)
        twopinListCombo = []
        twopinListComboCleared = []
        for i in range(len(init.gridParameters(grid_info)['netInfo'])):
            netNum = i
            netPinList = []; netPinCoord = []
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

            twoPinListVanilla = twoPinList

            # Insert Tree method to decompose two pin problems here
            twoPinList = tree.generateMST(twoPinList)
    #        print('Two pin list after:', twoPinList, '\n')

            # Remove pin pairs that are in the same grid 
            nullPairList = []
            for i in range(len(twoPinListVanilla)):
                if twoPinListVanilla[i][0][:3] == twoPinListVanilla[i][1][:3]:
                    nullPairList.append(twoPinListVanilla[i])
            for i in range(len(nullPairList)):
                twoPinListVanilla.reomove(nullPairList[i])

            # Remove pin pairs that are in the same grid 
            nullPairList = []
            for i in range(len(twoPinList)):
                if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                    nullPairList.append(twoPinList[i])
            for i in range(len(nullPairList)):
                twoPinList.reomove(nullPairList[i])

            # Key: use original sequence of two pin pairs
            twopinListComboCleared.append(twoPinListVanilla)

        # print('twopinListComboCleared',twopinListComboCleared)
        twoPinEachNetClear = []
        for i in twopinListComboCleared:
            num = 0
            for j in i:
                num = num + 1
            twoPinEachNetClear.append(num)

        # print('twoPinEachNetClear',twoPinEachNetClear)

        for i in range(len(init.gridParameters(grid_info)['netInfo'])):
            netNum = int(sortedHalfWireLength[i][0]) # i 
            # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
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
    #        print('Two pin list after:', twoPinList, '\n')

            # Remove pin pairs that are in the same grid 
            nullPairList = []
            for i in range(len(twoPinList)):
                if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                    nullPairList.append(twoPinList[i])

            for i in range(len(nullPairList)):
                twoPinList.reomove(nullPairList[i])

            # Key: Use MST sorted pin pair sequence under half wirelength sorted nets
            twopinListCombo.append(twoPinList)

        # print('twopinListCombo',twopinListCombo)

        # for i in range(1):
        for i in range(len(init.gridParameters(grid_info)['netInfo'])):

            # Determine nets to wire based on sorted nets (stored in list sortedHalfWireLength)
    #        print('*********************')
            # print('Routing net No.',init.gridParameters(grid_info)['netInfo'][int(sortedHalfWireLength[i][0])]['netName'])
            # (above output is to get actual netName)
    #        print('Routing net No.',sortedHalfWireLength[i][0])

            netNum = int(sortedHalfWireLength[i][0])

            # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
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

            # Remove pin pairs that are in the same grid 
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
                route, cost = twoPinASearch.AStarSearchRouter(pinStart, pinEnd, gridGraphSearch)
                routeListSingleNet.append(route)
                i += 1

            mergedrouteListSingleNet = []

            for list in routeListSingleNet:
                # if len(routeListSingleNet[0]) == 2:
                #     mergedrouteListSingleNet.append(list[0])
                #     mergedrouteListSingleNet.append(list[1])
                # else:
                for loc in list:
                        if loc not in mergedrouteListSingleNet:
                            mergedrouteListSingleNet.append(loc)

            routeListMerged.append(mergedrouteListSingleNet)
            routeListNotMerged.append(routeListSingleNet)

            # Update capacity and grid graph after routing one pin pair
            # # WARNING: there are some bugs in capacity update
            # # # print(route)
            # capacity = graph.updateCapacity(capacity, mergedrouteListSingleNet)
            # gridGraph = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

        # print('\nRoute List Merged:',routeListMerged)

        twopinlist_nonet = []
        for net in twopinListCombo:
        # for net in twopinListComboCleared:
            for pinpair in net:
                twopinlist_nonet.append(pinpair)

        # Get two pin numbers
        twoPinNum = 0
        twoPinNumEachNet = []
        for i in range(len(init.gridParameters(grid_info)['netInfo'])):
            netNum = int(sortedHalfWireLength[i][0]) # i
            twoPinNum = twoPinNum + (init.gridParameters(grid_info)['netInfo'][netNum]['numPins'] - 1)
            twoPinNumEachNet.append(init.gridParameters(grid_info)['netInfo'][netNum]['numPins'] - 1)

        # print('twoPinNumEachNet debug1: ',twoPinNumEachNet)
        # print('twopinlist_nonet',twopinlist_nonet)


        # DRL Module from here
        graphcase.max_step = 100 #20
        graphcase.twopin_combo = twopinlist_nonet
        # print('twopinlist_nonet',twopinlist_nonet)
        graphcase.net_pair = twoPinNumEachNet

        # Setting the session to allow growth, so it doesn't allocate all GPU memory.
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        sess = tf.Session(config=config)

        # Setting this as the default tensorflow session.
        keras.backend.tensorflow_backend.set_session(sess)

        # You want to create an instance of the DQN_Agent class here, and then train / test it.
        model_path = '../model/'
        data_path = '../data/'
        environment_name = 'grid'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        agent = DQN_Implementation.DQN_Agent(environment_name, sess, graphcase)   
             
        # Burn in with search 
        # Get a list of (observation, action, reward, observation_next, is_terminal) 
        # with Route List Merged (solution given with A*search plus tree)
        graphcaseBurnIn = graph.GridGraph(init.gridParameters(grid_info))
        graphcaseBurnIn.max_step = 10000

        observationCombo = []; actionCombo = []; rewardCombo = []
        observation_nextCombo = []; is_terminalCombo = []
        
        for enumerator in range(300):
            for i in range(gridParameters['numNet']):
                goal = routeListMerged[i][-1]
                graphcaseBurnIn.goal_state = (goal[3],goal[4],goal[2],goal[0],goal[1])
                for j in range(len(routeListMerged[i])-1):
                    position = routeListMerged[i][j]
                    nextposition = routeListMerged[i][j+1]
                    graphcaseBurnIn.current_state = (position[3],position[4],
                        position[2],position[0],position[1])
                    # print(graphcaseBurnIn.state2obsv())
                    observationCombo.append(graphcaseBurnIn.state2obsv())
                    action = graph.get_action(position,nextposition)
                    # print('action',action)
                    actionCombo.append(action)

                    graphcaseBurnIn.step(action)
                    rewardCombo.append(graphcaseBurnIn.instantreward)
                    # graphcaseBurnIn.current_state = (nextposition[3],nextposition[4],
                    #     nextposition[2],nextposition[0],nextposition[1])
                    observation_nextCombo.append(graphcaseBurnIn.state2obsv())
                    is_terminalCombo.append(False)

                is_terminalCombo[-1] = True

    # if testing using training function, comment burn_in

        agent.replay = DQN_Implementation.Replay_Memory() #Remove memeory of previous training

        agent.burn_in_memory_search(observationCombo,actionCombo,rewardCombo,
            observation_nextCombo,is_terminalCombo)

        twoPinNum = 0
        twoPinNumEachNet = []
        for i in range(len(init.gridParameters(grid_info)['netInfo'])):
            twoPinNum = twoPinNum + (init.gridParameters(grid_info)['netInfo'][i]['numPins'] - 1)
            twoPinNumEachNet.append(init.gridParameters(grid_info)['netInfo'][i]['numPins'] - 1)
        
        # print ("Two pin num: ",len(graphcase.twopin_combo))
        twoPinNum = len(graphcase.twopin_combo)
        twoPinNumEachNet  = graphcase.twopin_combo

        # Training DRL
        savepath = model_path #32by32simple_model_train"
        # model_file = "../32by32simple_model_train/model_24000.ckpt"

        # print ('twoPinNumEachNet debug2',twoPinNumEachNet)

    #         graphcase.max_step = 50 #20
    # graphcase.twopin_combo = twopinlist_nonet
    # graphcase.net_pair = twoPinNumEachNet

        # Reinitialze grid graph parameters before training on new benchmarks
        agent.gridParameters = gridParameters
        agent.gridgraph.max_step = 100
        agent.goal_state = None; agent.init_state = None
        agent.gridgraph.capacity = capacity
        agent.gridgraph.route  = []
        agent.gridgraph.twopin_combo = twopinlist_nonet
        agent.gridgraph.twopin_pt = 0
        agent.gridgraph.twopin_rdn = None
        agent.gridgraph.reward = 0.0
        agent.gridgraph.instantreward = 0.0
        agent.gridgraph.best_reward = 0.0
        agent.gridgraph.best_route = []
        agent.gridgraph.route_combo = []
        agent.gridgraph.net_pair = twoPinEachNetClear
        agent.gridgraph.instantrewardcombo = []
        agent.gridgraph.net_ind = 0
        agent.gridgraph.pair_ind = 0
        agent.gridgraph.posTwoPinNum = 0
        agent.gridgraph.passby = np.zeros_like(capacity)
        agent.previous_action = -1

        # print('twopinlist_nonet',twopinlist_nonet)
        episodes = agent.max_episodes
        # print('twopinlist_nonet',twopinlist_nonet)
        solution_combo_filled,reward_plot_combo,reward_plot_combo_pure,solutionTwoPin,posTwoPinNum \
        = agent.train(twoPinNum,twoPinEachNetClear,netSort,savepath,model_file=None)
       
        # pkl.dump(solution_combo_filled,fileObject)
        # fileObject.close()
        twoPinListPlotRavel = []
        for i in range(len(twopinlist_nonet)):
                twoPinListPlotRavel.append(twopinlist_nonet[i][0])
                twoPinListPlotRavel.append(twopinlist_nonet[i][1])


        # Generate output file for DRL solver
        # print('posTwoPinNum',posTwoPinNum)
        # print('len(graphcase.twopin_combo)',len(graphcase.twopin_combo))

        if posTwoPinNum >= len(graphcase.twopin_combo): #twoPinNum: 
            # Plot reward and save reward data
            n = np.linspace(1,episodes,len(reward_plot_combo))
            plt.figure()
            plt.plot(n,reward_plot_combo)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.savefig('test_benchmark_{dumpBench}.DRLRewardPlot.jpg'.format(dumpBench=globali+1))
            # plt.show()
            plt.close()

            n = np.linspace(1,episodes,len(reward_plot_combo_pure))
            plt.figure()
            plt.plot(n,reward_plot_combo_pure)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.savefig('test_benchmark_{dumpBench}.DRLRewardPlotPure.jpg'.format(dumpBench=globali+1))
            plt.close()
           

            filenameplot = '%s.rewardData' % filename
            np.save(filenameplot,reward_plot_combo)

            # dump solution of DRL
            f = open('solutionsDRL/test_benchmark_{dumpBench}.gr.DRLsolution'.format(dumpBench=globali+1), 'w+')
            # for i in range(1):
            twoPinSolutionPointer = 0
            routeListMerged = solution_combo_filled
            # print('solution_combo_filled',solution_combo_filled)
            for i in range(gridParameters['numNet']):
                singleNetRouteCache = []
                singleNetRouteCacheSmall = []
                indicator = i
                netNum = int(sortedHalfWireLength[i][0]) # i 
                
                i = netNum

                value = '{netName} {netID} {cost}\n'.format(netName=gridParameters['netInfo'][indicator]['netName'],
                                                      netID = gridParameters['netInfo'][indicator]['netID'],
                                                      cost = 0) #max(0,len(routeListMerged[indicator])-1))
                f.write(value)
                for j in range(len(routeListMerged[indicator])):
                    for  k in range(len(routeListMerged[indicator][j])-1):

                        a = routeListMerged[indicator][j][k]
                        b = routeListMerged[indicator][j][k+1]

                        if (a[3],a[4],a[2],b[3],b[4],b[2]) not in singleNetRouteCache:  
                        # and (b[3],b[4],b[2]) not in singleNetRouteCacheSmall:
                            singleNetRouteCache.append((a[3],a[4],a[2],b[3],b[4],b[2]))
                            singleNetRouteCache.append((b[3],b[4],b[2],a[3],a[4],a[2]))
                            # singleNetRouteCacheSmall.append((a[3],a[4],a[2]))
                            # singleNetRouteCacheSmall.append((b[3],b[4],b[2]))

                            diff = [abs(a[2]-b[2]),abs(a[3]-b[3]),abs(a[4]-b[4])]
                            if diff[1] > 2 or diff[2] > 2:
                                continue
                            elif diff[1] == 2 or diff[2] == 2:
                                continue
                            elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                                continue
                            elif diff[0] + diff[1] + diff[2] >= 2:
                                continue
                            else:
                                value = '({},{},{})-({},{},{})\n'.format(int(a[0]),int(a[1]),a[2],int(b[0]),int(b[1]),b[2])
                                f.write(value)
                    twoPinSolutionPointer = twoPinSolutionPointer + 1
                f.write('!\n')
            f.close()
            
            # Plot of routing for multilple net (RL solution) 3d
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlim(0.75,2.25)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            x_meshP = np.linspace(0,gridParameters['gridSize'][0]-1,200)
            y_meshP = np.linspace(0,gridParameters['gridSize'][1]-1,200)
            z_meshP = np.linspace(1,2,200)
            x_mesh,y_mesh = np.meshgrid(x_meshP,y_meshP)
            z_mesh = np.ones_like(x_mesh)
            ax.plot_surface(x_mesh,y_mesh,z_mesh,alpha=0.3,color='r')
            ax.plot_surface(x_mesh,y_mesh,2*z_mesh,alpha=0.3,color='r')
            plt.axis('off')
            plt.close()
            # for i in twoPinListPlotRavel:
            #     x = [coord[0] for coord in twoPinListPlotRavel]
            #     y = [coord[1] for coord in twoPinListPlotRavel]
            #     z = [coord[2] for coord in twoPinListPlotRavel]
            #     ax.scatter(x,y,z,s=25,facecolors='none', edgecolors='k')

            # Visualize solution
            for twoPinRoute in solutionTwoPin:
                x = []; y = []; z = []
                for i in range(len(twoPinRoute)):
                    # print(routeList[i])
                    # diff = [abs(routeList[i][2]-routeList[i+1][2]),
                    # abs(routeList[i][3]-routeList[i+1][3]),
                    # abs(routeList[i][4]-routeList[i+1][4])]
                    # if diff[1] > 2 or diff[2] > 2:
                    #     continue
                    # elif diff[1] == 2 or diff[2] == 2:
                    #     # print('Alert')
                    #     continue
                    # elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                    #     continue
                    # elif diff[0] + diff[1] + diff[2] >= 2:
                    #     continue
                    # else:
                    x.append(twoPinRoute[i][3])
                    y.append(twoPinRoute[i][4])
                    z.append(twoPinRoute[i][2])
                    ax.plot(x,y,z,linewidth=2.5)
            
            plt.xlim([0, gridParameters['gridSize'][0]-1])
            plt.ylim([0, gridParameters['gridSize'][1]-1])
            plt.savefig('DRLRoutingVisualize_test_benchmark_{dumpBench}.png'.format(dumpBench=globali+1))
            # plt.show()
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
            plt.savefig('DRLRoutingVisualize_test_benchmark2d_{dumpBench}.png'.format(dumpBench=globali+1))
            plt.close()

        else:
            print("DRL fails with existing max episodes! : (")
    except IndexError:
        print ("Invalid Benchmarks! ")
        agent.sess.close()
        tf.reset_default_graph()
    # graphcase.posTwoPinNum = 0
    return

def DRLagent_generator(filename):
    grid_info = init.read(filename)
    for item in init.gridParameters(grid_info).items():
        print(item)
    gridParameters = init.gridParameters(grid_info)

    # # GridGraph
    graphcase = graph.GridGraph(init.gridParameters(grid_info))
    capacity = graphcase.generate_capacity()
    gridX,gridY,gridZ = graphcase.generate_grid()
    # Real Router for Multiple Net
    # Note: pinCoord input as absolute length coordinates
    gridGraphSearch = twoPinASearch.AStarSearchGraph(gridParameters, capacity)
    # Sort net
    halfWireLength = init.VisualGraph(init.gridParameters(grid_info)).bounding_length()

    sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=True) # Large2Small
    netSort = []
    for i in range(gridParameters['numNet']):
        order = int(sortedHalfWireLength[i][0])
        netSort.append(order)

    routeListMerged = []
    routeListNotMerged = []

    # Getting two pin list combo (For RL)
    twopinListCombo = []
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        netNum = i
        netPinList = []; netPinCoord = []
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

#        print('Two pin list vanilla:',twoPinList,'\n')

        twoPinListVanilla = twoPinList

        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)
        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinListVanilla)):
            if twoPinListVanilla[i][0][:3] == twoPinListVanilla[i][1][:3]:
                nullPairList.append(twoPinListVanilla[i])

        for i in range(len(nullPairList)):
            twoPinListVanilla.reomove(nullPairList[i])
        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])

        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        # Key: use original sequence of two pin pairs
        # twopinListCombo.append(twoPinListVanilla)

    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        
        netNum = int(sortedHalfWireLength[i][0]) # i 
        # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
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

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])
        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])
        # Key: Use MST sorted pin pair sequence under half wirelength sorted nets
        twopinListCombo.append(twoPinList)

    # for i in range(1):
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):

        # Determine nets to wire based on sorted nets (stored in list sortedHalfWireLength)
        # print('Routing net No.',init.gridParameters(grid_info)['netInfo'][int(sortedHalfWireLength[i][0])]['netName'])
        # (above output is to get actual netName)
        netNum = int(sortedHalfWireLength[i][0])

        # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
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
        # Remove pin pairs that are in the same grid 
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
            route, cost = twoPinASearch.AStarSearchRouter(pinStart, pinEnd, gridGraphSearch)
            routeListSingleNet.append(route)
            i += 1
        mergedrouteListSingleNet = []

        for list in routeListSingleNet:
            # if len(routeListSingleNet[0]) == 2:
            #     mergedrouteListSingleNet.append(list[0])
            #     mergedrouteListSingleNet.append(list[1])
            # else:
            for loc in list:
                    if loc not in mergedrouteListSingleNet:
                        mergedrouteListSingleNet.append(loc)
        routeListMerged.append(mergedrouteListSingleNet)
        routeListNotMerged.append(routeListSingleNet)

    twopinlist_nonet = []
    for net in twopinListCombo:
        for pinpair in net:
            twopinlist_nonet.append(pinpair)
    # Get two pin numbers
    twoPinNum = 0
    twoPinNumEachNet = []
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        netNum = int(sortedHalfWireLength[i][0]) # i
        twoPinNum = twoPinNum + (init.gridParameters(grid_info)['netInfo'][netNum]['numPins'] - 1)
        twoPinNumEachNet.append(init.gridParameters(grid_info)['netInfo'][netNum]['numPins'] - 1)

    graphcase.max_step = 50 #20
    graphcase.twopin_combo = twopinlist_nonet
    graphcase.net_pair = twoPinNumEachNet

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    model_path = '../model/'
    data_path = '../data/'
    environment_name = 'grid'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    agent = DQN_Implementation.DQN_Agent(environment_name, sess, graphcase)
    episodes = agent.max_episodes
    return agent,episodes,model_path,data_path,environment_name

if __name__ == "__main__":
    os.makedirs('solutionsDRL')
    reduced_path = 'benchmark_reduced'
    if not os.path.exists(reduced_path):
        os.makedirs(reduced_path)

    list = os.listdir('benchmark_reduced')
    benchmarkNum = len(list)

    
    # agent,episodes,model_path,data_path,environment_name = DRLagent_generator(filename)

    for i in range(benchmarkNum):
        filename = "benchmark_reduced/test_benchmark_{num}.gr".format(num=i+1)
        print ('******************************\n')
        print ('Working on {bm}\n'.format(bm=filename))
        DRL_implementation(filename,i)






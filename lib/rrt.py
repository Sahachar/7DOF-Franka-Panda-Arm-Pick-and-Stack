from typing import Type
import numpy as np
import random as rnd

from numpy.lib.polynomial import polyder
from lib.detectCollision import detectCollision         #lib
from lib.detectCollision import detectCollisionOnce     #lib
from lib.loadmap import loadmap                         #lib
from copy import deepcopy
from math import pi
from lib.calculateFK import FK                          #lib
from collections import namedtuple
from lib.solveIK import IK


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    #instantiate FK()
    fk = FK()
    ik = IK()
    # initialize path
    path = []

    # initialize Tstart_node, Tstart_edge, and Tgoal_node, Tgoal_edge
    # Tstart_node is a list of nodes, where each node is a (7,) np array
    # Tstart_edge is a list of edges, where each edge is a vector from closest node to random node
    #print(start, start.shape)
    Tstart_node = []
    Tstart_edge = []
    Tstart_node.append(start)
    Tstart_edge.append(0)

    joint_positions_start, start_homo = fk.forward(start)
    tree_start = []
    tree_start.append(start_homo)

    Tgoal_node = []
    Tgoal_edge = []
    Tgoal_node.append(goal)
    Tgoal_edge.append(0)

    joint_positions_goal, goal_homo = fk.forward(goal)
    tree_goal = []
    tree_goal.append(goal_homo)


    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])


    #rnd.seed(1)

    finished = False
    start_bool = False
    goal_bool = False
    volume = 0.05
    safe_volume = 0.001
    start = True
    obstacles = map.obstacles
    k=0
    while(finished == False and k < 2000):
        k=k+1
        if(k%1000)==0:
            print(k)
        #initialize random q configuration within limits
        q_random = np.zeros((7,))

        accept_q = False

        while(accept_q == False):
            #print(accept_qt)
            for joint in range(len(q_random)):
                q_random[joint] = rnd.uniform(lowerLim[joint], upperLim[joint])
        #print('q_random', q_random)
        #find end effector homog transformation T0e
            joint_positions, Tra = fk.forward(q_random)
            accept_q = check_self_collision(joint_positions,Tra,volume,safe_volume,start)
            for obstacle in obstacles:
                accept_q = check_collision_links(joint_positions,Tra,obstacle,True)
                #print(accept_q,joint_positions)
        increment_movement=20
        for obstacle in obstacles:
            start_bool = False
            point1, point2, closest_node,clos_homog = evaluate_closest_q(Tra, tree_start)
        #this only checks collisions for the end effector trajectory.
            q_closest_node, q_intermediate = intermediate_and_closest_q(Tstart_node, closest_node, q_random, increment_movement)
            #print('q_closest_node',q_closest_node)
            joint_positions, T0e = fk.forward(q_intermediate)
            for obstacle in obstacles:
                accept_q = check_collision_links(joint_positions,T0e,obstacle,True)

            if (check_collision_configurations(q_closest_node,q_intermediate,obstacle) == False and accept_q == True):
                #Check collision robot against obstacle
                Tstart_node.append(q_intermediate)
                Tstart_edge.append(closest_node)
                q_inter_joint_pos, inter_homo = fk.forward(q_intermediate)
                tree_start.append(inter_homo)
                start_bool = True
                #print('Tstart_node', Tstart_node)
                #print('start_bool is', start_bool)
                #print('collide')
                #print('Tstart_node', Tstart_node)
                #print('Tstart_edge', Tstart_edge)
                #print('tree_start', tree_start, tree_start[0].shape)

            else:
                start_bool = False

            goal_bool = False
            point1, point2, closest_node, clos_homog = evaluate_closest_q(Tra, tree_goal)
            q_closest_node, q_intermediate = intermediate_and_closest_q(Tgoal_node, closest_node, q_random, increment_movement)
            #print('q_closest_node',q_closest_node)
            joint_positions, T0e = fk.forward(q_intermediate)
            for obstacle in obstacles:
                accept_q = check_collision_links(joint_positions,T0e,obstacle,True)

            if (check_collision_configurations(q_closest_node,q_intermediate,obstacle) == False and accept_q == True):
                Tgoal_node.append(q_intermediate)
                Tgoal_edge.append(closest_node)
                q_inter_joint_pos, inter_homo = fk.forward(q_intermediate)
                tree_goal.append(inter_homo)
                goal_bool = True
                #print('Tgoal_node', Tgoal_node)
                #print('goal_bool is', goal_bool)
            else:
                goal_bool = False


        finished = start_bool and goal_bool

        if (finished == True):
            Tstart_node.append(q_random)
            Tstart_edge.append(len(Tstart_node)-2)
            Tgoal_node.append(q_random)
            Tgoal_edge.append(len(Tgoal_node)-2)


    #print('tree_start', Tstart_node,Tstart_edge)
    #print('tree_goal', Tgoal_node,Tgoal_edge)
            #print('collide')
            #print('Tstart_node', Tgoal_node)
            #print('Tstart_edge', Tgoal_edge)
            #print('tree_start', tree_goal, tree_goal[0].shape)
        #print('finished',finished)
        #print('jointpos', joint_positions, 'T0e', T0e)
    print('tree_start', Tstart_node, Tstart_edge)
    print('tree_goal', Tgoal_node, Tgoal_edge)


#check for collisions with multiple blocks
#collisions with link (line), itself
#collisions with blocks
#collision for the q_random at the beginning
#create path
    #path.append(Tgoal_node[-1])
    #print(len(Tgoal_node))
    #print(Tgoal_node[Tgoal_edge[-2]])
    #path.append(Tgoal_node[Tgoal_edge[-2]])
    path_start_reverse = create_path(Tstart_node, Tstart_edge)
    print("path_start_inverse",path_start_reverse)
    path_start_reverse.reverse()

    path_goal = create_path(Tgoal_node, Tgoal_edge)
    print('path_goal',path_goal)
    path.append(path_start_reverse)
    path.append(path_goal)
    #path = np.array(path)
    path = np.concatenate(path, axis=0)
    print('final path', path.shape)
    return path

def create_path(Tstart_node, Tstart_edge):
    pos = 2000
    out = 2000
    path = []
    i = 0
    while(pos != 0):
        if(i == 0):
            path.append(Tstart_node[len(Tstart_edge)-1])
            pos = Tstart_edge[len(Tstart_node)-1]
            i=i+1
        else:
            path.append(Tstart_node[pos])
            pos = Tstart_edge[pos]
            i = i + 1
        #print(pos)
    path.append(Tstart_node[0])

    return path

def add_node(node_i, node_all):
    node_all.append(node_i)
    return

def remove_node(node_i, node_all): #consider pop vs del vs remove. Pop returns value, so good for checking.
    deleted_node = node_all.pop(node_number)
    return deleted_node

def add_edge(parent, child):
    return

def remove_edge():
    return

def print_number_nodes_all():
    num_node_all = len(node_all)
    return num_node_all

def check_collision_configurations(q1,q2,obstacle):
    fk=FK()
    start_points = []
    finish_points = []
    joint_positions1, T1e = fk.forward(q1)
    joint_positions2, T2e = fk.forward(q2)

    collide = False
    #record all the data points of the joints for new configuration
    for i in range (7):
        start_points.append([joint_positions1[i,0],
                             joint_positions1[i,1],
                             joint_positions1[i,2]])
        finish_points.append([joint_positions2[i,0],
                             joint_positions2[i,1],
                             joint_positions2[i,2]])

    start_points.append([T1e[0,3],
                         T1e[1,3],
                         T1e[2,3]])
    finish_points.append([T2e[0,3],
                         T2e[1,3],
                         T2e[2,3]])

    start_points_array = np.asarray(start_points)
    finish_points_array = np.asarray(finish_points)
    #print('s_points', start_points)
    #print('reshaped_s', start_points_array, start_points_array.shape)
    #print('f_points', finish_points)

    #print('here')
    if detectCollision(start_points_array,finish_points,obstacle) == True:
        collide=True
        #print('collide')
    return collide




def check_collision_links(joint_positions,T0e,obstacle,start):
    point = []
    accept_qt = np.zeros((7), dtype=bool)
    #save all the data points of the joints for new configuration
    for i in range (7):
        point.append([joint_positions[i,0],
                      joint_positions[i,1],
                       joint_positions[i,2]])
    point.append([T0e[0,3],
                  T0e[1,3],
                  T0e[2,3]])

    for i in range (7):
        if start == True:
            start_position = np.delete(point,7,0)
            finish_position = np.delete(point,0,0)
            start = False
        else:
            start_position=np.delete(start_position,0,0)
            finish_position=np.delete(finish_position,0,0)

        #Check the pose of the robot against a given obstacle
        if detectCollisionOnce(start_position[0],finish_position[0],obstacle) == True:
            acept_q = False
            accept_qt[i]=False
            #print('aqui')
        else:
            accept_qt[i]=True
            #print(start_position)
            #print(finish_position)
            #print('aquí else')
            #print(len(start_position))
            #print(accept_qt)
    accept_q = accept_qt[0] & accept_qt[1] & accept_qt[2] & accept_qt[3] & accept_qt[4] & accept_qt[5] & accept_qt[6]
    #print(accept_q)
    return accept_q

def check_self_collision(joint_positions,T0e,volume,safe_volume,start):
    point = []
    accept_qt = np.zeros((6), dtype=bool)
    #save all the data points of the joints for new configuration
    for i in range (7):
        point.append([joint_positions[i,0],
                      joint_positions[i,1],
                       joint_positions[i,2]])
    point.append([T0e[0,3],
                  T0e[1,3],
                  T0e[2,3]])

    #check if the robot pose is coliding with itself
    #create boxes that represent each linkstart_position
    for i in range (6):
        ev_point1 = point[i]
        ev_point2 = point[i+1]
        if(ev_point1[0]==ev_point2[0]):
            boxpointxmin = ev_point1[0]-volume
            boxpointxmax = ev_point1[0]+volume
        else:
            boxpointxmin = ev_point1[0]+safe_volume
            boxpointxmax = ev_point2[0]-safe_volume
        if(ev_point1[1]==ev_point2[1]):
            boxpointymin = ev_point1[1]-volume
            boxpointymax = ev_point1[1]+volume
        else:
            boxpointymin = ev_point1[1]+safe_volume
            boxpointymax = ev_point2[1]-safe_volume
        if(ev_point1[2]==ev_point2[2]):
            boxpointzmin = ev_point1[2]-volume
            boxpointzmax = ev_point1[2]+volume
        else:
            boxpointzmin = ev_point1[2]+safe_volume
            boxpointzmax = ev_point2[2]-safe_volume
            box=[boxpointxmin,boxpointymin,boxpointzmin,boxpointxmax,boxpointymax,boxpointzmax]
            #print(box)
            # Check for pose validation in terms of self colition
        if start == True:
            start_position = np.delete(point,7,0)
            finish_position = np.delete(point,0,0)
            start = False
        else:
            start_position=np.delete(start_position,0,0)
            finish_position=np.delete(finish_position,0,0)

        #Check collision from each link against the an approximate volume of the preceding link
        if len(start_position) > 1:
            #print(i,len(start_position))
            for i in range (len(start_position)-1):
                #print(i,len(start_position))
                if detectCollisionOnce(start_position[i],finish_position[i],box) == True:
                    acept_q = False
                    #print('aquí')
                else:
                    accept_qt[len(start_position)-2]=True
                    #print(start_position)
                    #print(finish_position)
                    #print('aquí else')
            #print(len(start_position))
        accept_q = accept_qt[0] & accept_qt[1] & accept_qt[2] & accept_qt[3] & accept_qt[4] & accept_qt[5]
        #print(accept_q)
    return accept_q

def intermediate_and_closest_q(T_node, closest_node, q_random, increment_movement):
    #ik = IK()
    #seems sensitive to seed
    q_closest_node = T_node[closest_node]
    joint_differences = q_random - q_closest_node
    joint_incremental_movement = joint_differences / increment_movement

    #print('check this', joint_differences, joint_incremental_movement)
    #print('q_closest_node', q_closest_node)
    #print('q_rnd', q_random)
    q_intermediate = q_closest_node + joint_incremental_movement
    #print('joint_incremental_movement', joint_incremental_movement)
    #print('q_intermediate', q_intermediate)

    return q_closest_node, q_intermediate

def index_closest_node(q_rnd_homog, tree_homog):
#    distances = []
#    for node in range(len(tree_homo)):
#        distance = np.sqrt((rnd[0,3]-tree_homo[node,0,3])**2 +
#                           (rnd[1,3]-tree_homo[node,1,3])**2 +
#                           (rnd[2,3]-tree_homo[node,2,3])**2)
#       distances.append(distance)
#       min(distances)
    shortest_distance = 1e6
    pos_connection = 0

    for node in range(len(tree_homog)):
        tree_homog_node = tree_homog[node]
        distance = np.sqrt((q_rnd_homog[0,3]-tree_homog_node[0,3])**2 +
                           (q_rnd_homog[1,3]-tree_homog_node[1,3])**2 +
                           (q_rnd_homog[2,3]-tree_homog_node[2,3])**2)
        if(distance < shortest_distance):
            shortest_distance = distance
            pos_connection = node
    return pos_connection


def evaluate_closest_q(T0e, tree_initial):
    closest_node = index_closest_node(T0e, tree_initial)

    #print('closest_node', closest_node)
    clos_homog = tree_initial[closest_node]
    point1 = np.array([clos_homog[0,3],
                       clos_homog[1,3],
                       clos_homog[2,3]])

    point2 = np.array([T0e[0,3],
                       T0e[1,3],
                       T0e[2,3]])
    #print('point1', point1, point1.shape)
    #print('point1', [point1])
    return point1, point2, closest_node,clos_homog




if __name__ == '__main__':
    #map_struct = loadmap("/home/luigi/meam520_ws/src/meam520_labs/lib/maps/map1.txt")
    map_struct = loadmap("../maps/map1.txt")
    #print(map_struct.dtype, 'map_struct')
    #map_struct = loadmap("../maps/map_test.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))




    # matches figure in the handout


    print('path', path)
    #print('hi')

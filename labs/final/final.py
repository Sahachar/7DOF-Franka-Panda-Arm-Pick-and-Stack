import sys
import numpy as np
from copy import deepcopy

import rospy
import roslib

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
import time
import random


# The library you implemented over the course of this semester!
from lib.calculateFK import FK as FK1
from lib.calcJacobian import FK
from lib.solveIK import IK
from lib.rrt import rrt
from lib.loadmap import loadmap


lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

TAG_0 = ['tag0']
STATIC_TAGS = ['tag6', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
DYNAMIC_TAGS = ['tag7', 'tag8', 'tag9', 'tag10', 'tag11', 'tag12']

def joint_check(q):
    out_of_limits = False
    for i in range(7):
        if q[i] < lower[i]:
            out_of_limits = True
            break
        elif q[i] > upper[i]:
            out_of_limits = True
    return out_of_limits



def move_to(tag_any,tag0_cam,safety_distance, reorient):
    #   T1 is the Block pose
    #   T0 is the target 0 reference to calculate the final pose based on the robot based
    #   dz is a differenial of height to approach the objects
    H = np.zeros((4, 4))
    tag0_robot = np.array([[1, 0, 0, 0.5],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    rot_x = np.array([[1, 0, 0, 0],  #rotation around x with 180 to allign
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    rot_z = np.array([[0, -1, 0, 0],  #rotation around x with 180 to allign
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
#[WARN] [1638927244.756388, 5892.076000]: Position 3.63 for joint 6 violates joint limits [-2.89730,2.89730]. Constraining within range.


    #Get H from the robot to the targets

    H = np.matmul(np.linalg.inv(np.matmul(tag0_cam,tag0_robot)),tag_any)
    if (reorient == True):
        H = np.matmul(H,rot_x)
        H = np.matmul(H,rot_z)
    elif (reorient == False):
        H = np.matmul(H,rot_x)

    H[2,3] = H[2,3] + safety_distance
    #print(H)

    #distance, angle = ik.distance_and_angle(T1, T0)

    q, success, rollout = ik.inverse(H, arm.neutral_position())
    # print('\n q \n', q)

    #only checks for joint 6, should we check more?
    if (success == False):
        # print('\n IK solver returned joints out of limits. \n')
        while (q[6] > upper[6]):
            q[6] -= np.pi/2
        while (q[6] < lower[6]):
            q[6] += np.pi/2
        # print("Success: ",success, '\n new q' , q)
    else:
        print("Success: ",success, '\n q is safe now!' )
    #print("Solution: ",q)
    #print("Iterations:", len(rollout))

    arm.safe_move_to_position(q)

def place_block(tag, tagX_H, tag0_cam, block_width, tower_pos, colour):
    #   T1 is the object pose
    #   T0 is the target 0 reference to calculate the final pose based on the robot based
    #   dz is a differenial of height to approach the objects
    #   tow_pos is the x,y and z position of the tower

    half_block_width = 0.05 / 2
    safety = 0.02


    tag0_robot = np.array([[1, 0, 0, 0.5],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    rot_x = np.array([[1, 0, 0, 0],  #rotation of 180 to allign
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    rot_ym = np.array([[0, 0, -1, 0],  #rotation of 180 to allign
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
    #Get H from the robot to the targets
    H = np.matmul(np.linalg.inv(np.matmul(tag0_cam,tag0_robot)),tagX_H)
    H = np.matmul(H, rot_x)
    H[2,3] = tower_pos[2] + block_width + safety


    q1, success, rollout = ik.inverse(H, arm.neutral_position())
    # print("Success: ",success)

    if colour == 'blue':
        q1[0] = q1[0]-np.pi/7
        q1[6] = - np.pi/4
    else:
        q1[0] = q1[0]+np.pi/7
        q1[6] = - np.pi/4

    joint_positions, T0e = fk.forward(q1)

    T0e[0,3] = tower_pos[0]
    T0e[1,3] = tower_pos[1]
    T0e[2,3] = tower_pos[2] + (block_width + safety) *2



    if tag != 'tag6' and tag != 'tag5':
        T0e = np.matmul(T0e, rot_ym)
        q, success, rollout = ik.inverse(T0e, q1)
    else:
        q, success, rollout = ik.inverse(T0e, q1)
        q_safe = q


    if (success == False):
        # print('\n IK solver returned joints out of limits. \n')
        while (q[6] > upper[6]):
            q[6] -= np.pi/2
        while (q[6] < lower[6]):
            q[6] += np.pi/2
        # print("Success: ",success, '\n new q' )
    else:
        print("Success: ",success, '\n q is safe now!' )

    # double checks for safety of all the  q, assumes it sees tag6 if this is true
    # if (joint_check(q)):
    #     q = q_safe
    #     while (q[6] > upper[6]):
    #         q[6] -= np.pi/2
    #     while (q[6] < lower[6]):
    #         q[6] += np.pi/2

    # print("Success: ",success)
    arm.safe_move_to_position(q)
    # print('choose this as a first good q!!!!', q)
    #T0e[2,3] = T0e[2,3] - block_width - half_block_width
    T0e[2,3] = T0e[2,3] - block_width * 2
    q2, success, rollout = ik.inverse(T0e, q)
    # print('choose this as a second good q!!!!' , q2)
    # print("Success: ",success)
    arm.safe_move_to_position(q2)

    arm.open_gripper()

    arm.safe_move_to_position(q)

    #arm.safe_move_to_position(arm.neutral_position())

def pp_static_blocks(T, colour):


    block_width = 0.05
    safety_distance = 0.01
    half_block_width = block_width / 2
    hover_height = block_width * 2

    if colour == 'blue':
        platform_x = 0.55
        platform_y = -0.24
        platform_z = 0.265
    else:
        platform_x = 0.6
        platform_y = 0.2
        platform_z = 0.265


    #gripper_width_open = 0.06
    gripper_width_close = 0.03 # 0.049
    gripper_force = 30 # 0.2

    j = 0
    for tag in STATIC_TAGS:
        if tag in T.keys():
            for i in range(len(T[tag])):
                #Start the sequence to stack the static blocks (All the sequence can be get in a loop)
                tower_pos = ((platform_x, platform_y, platform_z + block_width* j))        #Final position for the tower (x,y,z), z has to change for each block
                #print('Moving towards block ', str(j))
                move_to(T[tag][i], T['tag0'][0], block_width + safety_distance, reorient = False)                               #Approach the end effector to the orientation and position of block T[1] with a dz of 0.05
                #print('Opening gripper to pick up block ', str(j))
                arm.open_gripper()                         #Open the gripper over the block
                # print('Opening gripper to pick up block ', str(j))
                # arm.exec_gripper_cmd(gripper_width_open, gripper_force)                          #Open the gripper over the block
                #print('Moving down to pick up block', str(j))
                move_to(T[tag][i], T['tag0'][0], -half_block_width, reorient = False)                  #Takes the end effector to the heigh of the block in order to pick it up
                #print('Closing gripper to pick up block', str(j))
                arm.exec_gripper_cmd(gripper_width_close, gripper_force)                         #Close the gripper
                #print('Moving up')
                move_to(T[tag][i], T['tag0'][0], hover_height, reorient = False )                  #Picks up the gripper and moves to the same x,y position of the block, but put it 0.1m over
                #print('Releasing block')
                place_block(tag, T[tag][i], T['tag0'][0], block_width, tower_pos, colour)                  #Complete cycle to stack the block, tow_pos has to be updated each time a block is stacked
                j += 1
    #This is the same sequence as before, the difference is the block T[2] and the tow_pos

def pick_reorient_place_static(T, colour):

    block_width = 0.05
    safety_distance = 0.01
    half_block_width = block_width / 2
    hover_height = block_width * 2

    if colour == 'blue':
        platform_x = 0.55
        platform_y = -0.24
        platform_z = 0.2
    else:
        platform_x = 0.6
        platform_y = 0.22
        platform_z = 0.2

    #gripper_width_open = 0.06
    gripper_width_close = 0.03 # 0.049
    gripper_force = 30 # 0.2

    j = 0
    for tag in STATIC_TAGS:
        if tag in T.keys():
            for i in range(len(T[tag])):
                #Start the sequence to stack the static blocks (All the sequence can be get in a loop)
                tower_pos = ((platform_x, platform_y, platform_z + block_width * j + safety_distance * 2))        #Final position for the tower (x,y,z), z has to change for each block
                #print('Moving towards block ', str(j))
                move_to(T[tag][i], T['tag0'][0], block_width + safety_distance, reorient = True)                               #Approach the end effector to the orientation and position of block T[1] with a dz of 0.05

                #print('Opening gripper to pick up block ', str(j))
                arm.open_gripper()                         #Open the gripper over the block
                # print('Opening gripper to pick up block ', str(j))
                # arm.exec_gripper_cmd(gripper_width_open, gripper_force)                          #Open the gripper over the block
                #print('Moving down to pick up block', str(j))
                move_to(T[tag][i], T['tag0'][0], -half_block_width, reorient = True)                  #Takes the end effector to the heigh of the block in order to pick it up
                #print('Closing gripper to pick up block', str(j))
                arm.exec_gripper_cmd(gripper_width_close, gripper_force)    #Close the gripper
                #print('Moving up')
                if team == 'blue':
                    move_to(T[tag][i], T['tag0'][0], block_width * 2, reorient = True)           #Picks up the gripper and moves to the same x,y position of the block, but put it 0.15m over
                if team == 'red':
                    move_to(T[tag][i], T['tag0'][0], block_width * 3, reorient = True)           #Picks up the gripper and moves to the same x,y position of the block, but put it 0.15m over

                move_to(T[tag][i], T['tag0'][0], hover_height, reorient = True )                  #Picks up the gripper and moves to the same x,y position of the block, but put it 0.1m over
                #print('Releasing block')
                place_block(tag, T[tag][i], T['tag0'][0], block_width, tower_pos, colour)                  #Complete cycle to stack the block, tow_pos has to be updated each time a block is stacked
                j += 1

def test_standby_pose(colour):

    arm.open_gripper()
    if colour == 'red':
        standby = [ np.pi/2-0.1, 0.45,  0.20026807, -1.6, -.25,  np.pi/2+0.5, 2.5]
        q_hover = [ 1.48579024,  0.53478796,  0.19272007, -1.46678863, -0.2561729,   2.02234591,  2.4920735 ]
        q_hover_closer = [ 1.51083986,  0.56869837,  0.15558379, -1.51355154, -0.25108088,  2.10595084,  2.49886902]
    else:
        standby = [ np.pi/2-0.1-np.pi, 0.45,  0.20026807, -1.6, -.25,  np.pi/2+0.5, 2.5]
        q_hover = [ 1.48579024-np.pi,  0.53478796,  0.19272007, -1.46678863, -0.2561729,   2.02234591,  2.4920735 ]
        q_hover_closer = [ 1.51083986 -np.pi,  0.56869837,  0.15558379, -1.51355154, -0.25108088,  2.10595084,  2.49886902]

    # q_hover safe enough.
    # arm.safe_move_to_position(standby)
    # print('standby done')
    arm.safe_move_to_position(q_hover)
    # print('hover done')
    arm.safe_move_to_position(q_hover_closer)


def test_open_close(colour, dynamics_picked):
    if team == 'red':
        q_to_platform = [ 0.12919579-0.21,  0.28422413,  0.21319535, -1.77354599, -0.06693798,  2.0507466, -0.66594934]
        q_drop_off = [ 0.11613111-0.21,  0.36787458,  0.2247942,  -1.85668544, -0.10032638,  2.21366911,  -0.64435884]
    else:
        q_to_platform = [ 0.12919579-0.5,  0.28422413,  0.21319535, -1.77354599, -0.06693798,  2.0507466, -0.66594934]
        q_drop_off = [ 0.11613111-0.5,  0.36787458,  0.2247942,  -1.85668544, -0.10032638,  2.21366911,  -0.64435884]
    dz = 0.05
    #gripper_width_open = 0.06 will use open_gripper() instead
    gripper_width_close = 0.03 #0.049
    gripper_force = 30
    block_width = 0.05
    safety = 0.02
    start_time_pulse = time_in_seconds()
    # print('start_time' ,start_time_pulse - time_in_seconds())


    arm.exec_gripper_cmd(gripper_width_close)
    # print('FIRST gripper_width', gripper_width_close)


    for j in range(50):
        arm.open_gripper()

        start_time = time_in_seconds()
        time_current = time_in_seconds()
        random_time = random.uniform(0.5, 2.0)
        while (time_current - start_time < random_time): #this might need fine-tuning
            time_current = time_in_seconds()
        # print('used open_gripper()')

        #print('time', random_time)
        #CAN TEST TO SEE IF WE NEED A DELAY
        """
        start_time = time_in_seconds()
        time_current = time_in_seconds()
        while (time_current - start_time < 0.1):
            time_current = time_in_seconds()
        """
        state_dictionary = arm.get_gripper_state()
        # print('actual opening distance between gripper arms', state_dictionary['position'][0]+state_dictionary['position'][0])

        arm.exec_gripper_cmd(gripper_width_close, gripper_force)
        # print('gripper_width commanded', gripper_width_close)
        # print('time', time_in_seconds() - start_time_pulse)



        start_time = time_in_seconds()
        time_current = time_in_seconds()
        while (time_current - start_time < 0.5): #this might need fine-tuning
            time_current = time_in_seconds()


        state_dictionary = arm.get_gripper_state()
        # print('actual closing distance between gripper arms', state_dictionary['position'][0]+state_dictionary['position'][1])
        # print('gripper distance 1', state_dictionary['position'][0])
        # print('gripper distance 2', state_dictionary['position'][1])
        # print('gripper force 1', state_dictionary['force'][0])
        # print('gripper force 2', state_dictionary['force'][1])
        distance_between_gripper_arms = state_dictionary['position'][0]+state_dictionary['position'][0]

        if (distance_between_gripper_arms > 0.0495 or state_dictionary['force'][0] > 1 ):
            # print('TOUCHED UP DYNAMIC BLOCK!!!')
            arm.exec_gripper_cmd(gripper_width_close, gripper_force)
            # print('closing grippers one last time')
            break

    if dynamics_picked == 0:
        arm.safe_move_to_position(arm.neutral_position())
        arm.safe_move_to_position(q_to_platform)
        arm.safe_move_to_position(q_drop_off)
        arm.open_gripper()
    else:
        arm.safe_move_to_position(arm.neutral_position())
        joint_positions, T0e = fk.forward(q_to_platform)
        T0e[2,3] = T0e[2,3] + (block_width + safety/2) * dynamics_picked
        q_to_platform_new, success, rollout = ik.inverse(T0e, q_to_platform)
        arm.safe_move_to_position(q_to_platform_new)
        joint_positions, T0e = fk.forward(q_drop_off)
        T0e[2,3] = T0e[2,3] + (block_width + safety/2) * dynamics_picked
        q_drop_off_new, success, rollout = ik.inverse(T0e, q_drop_off)
        arm.safe_move_to_position(q_drop_off_new)
        arm.open_gripper()

    # print('after motion distance between gripper arms', state_dictionary['position'][0]+state_dictionary['position'][0])
    # print('close time', time_in_seconds() - start_time_pulse)
    # print('FINAL GRIPPER STATE', state_dictionary['position'][0]+state_dictionary['position'][0])

    return dynamics_picked

def save_tags_to_T_dictionary():

    T = {}
    # Save tags...

    for (tag, pose) in detector.get_detections():
        # print('\n', tag, '\n', pose)

        if tag in T.keys():
            T[tag].append(pose)
        else:
            T[tag] = [pose]

    # for tag in STATIC_TAGS:
    #     if tag in T.keys():
    #         print('\n tag static', tag, '\n', T[tag])
    return T

def find_closest_tag(T, DYNAMIC_TAGS):
    dist_smallest = 1e6
    T_robot = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])


    for tag in DYNAMIC_TAGS:
        if tag in T.keys():
            for i in range(len(T[tag])):
                distance, angle = ik.distance_and_angle(T[tag][i], T_robot)
                # print('distance for', tag, 'is:',  distance)
                if distance < dist_smallest:
                    dist_smallest = distance

                    i_smallest = i
                    tag_smallest = tag
                    # print('Tag which is closest to robot', tag, T[tag][i])

class DynamicBlocks:

    def __init__(self):
        self.poses = []

    def add_pose(self, new_pose):
        self.poses.append(new_pose)

if __name__ == "__main__":

    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    arm.safe_move_to_position(arm.neutral_position()) # on your mark!
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
        colour = 'blue'
    else:
        print("**  RED TEAM  **")
        colour = 'red'
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!


    # STUDENT CODE HERE
    ik = IK()
    fk = FK1()

    T = save_tags_to_T_dictionary()

    pick_reorient_place_static(T, colour)
    arm.safe_move_to_position(arm.neutral_position())
    dynamics_picked = 0
    for i in range(10):
        test_standby_pose(colour)
        dynamics_picked = test_open_close(colour, dynamics_picked) #only if standby pose seems safe
        arm.safe_move_to_position(arm.neutral_position())
        dynamics_picked += 1
    # Move around...
    #arm.safe_move_to_position(arm.neutral_position() + .1)

    # END STUDENT CODE

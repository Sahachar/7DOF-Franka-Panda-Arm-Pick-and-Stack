# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 00:33:27 2021

@author: alice
"""

import numpy as np
from math import pi

class FK():

    def __init__(self, l0 = 0.141, l1 = 0.192, l2 = 0.195,
                 l3 = 0.121, l4 = 0.083, l5 = 0.082,
                 l6 = 0.125, l7 = 0.259, l8 = 0.088,
                 l9 = 0.051, l10 = 0.159, l11 =0.015):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6
        self.l7 = l7
        self.l8 = l8
        self.l9 = l9
        self.l10 = l10
        self.l11 = l11
        pass

    def DHm(a,al,d,th):
        H = np.array([[np.cos(th),-np.sin(th)*np.cos(al),np.sin(th)*np.sin(al),a*np.cos(th)],
                     [np.sin(th),np.cos(th)*np.cos(al),-np.cos(th)*np.sin(al),a*np.sin(th)],
                     [0,np.sin(al),np.cos(al),d],
                     [0,0,0,1]])
        return H


    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions - 7 x 3 matrix, where each row corresponds to a rotational joint of the robot
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your code starts here
        H_10 = FK.DHm(0,0,self.l0,0)
        H_21 = FK.DHm(0,-pi/2,self.l1,q[0])
        H_321 = FK.DHm(0,pi/2,0,q[1])
        H_322 = FK.DHm(0,0,self.l2,0)
        H_43 = FK.DHm(self.l5,pi/2,self.l3,q[2])
        H_541 = FK.DHm(-self.l4,-pi/2,0,q[3])
        H_542 = FK.DHm(0,0,self.l6,0)
        H_651 = FK.DHm(0,pi/2,self.l7,q[4])
        H_652 = FK.DHm(0,0,-self.l11,q[5])
        H_761 = FK.DHm(0,0,self.l11,0)
        H_762 = FK.DHm(self.l8,pi/2,0,0)
        H_763 = FK.DHm(0,0,self.l9,0)
        H_87 = FK.DHm(0,0,self.l10,q[6]-pi/4)

        jointPositions = np.zeros((7,3))

        H_10 = H_10
        H_20 = np.matmul(H_10,H_21)
        H_31 = np.matmul(H_20,H_321)
        H_30 = np.matmul(H_31,H_322)
        H_40 = np.matmul(H_30,H_43)
        H_51 = np.matmul(H_40,H_541)
        H_50 = np.matmul(H_51,H_542)
        H_61 = np.matmul(H_50,H_651)
        H_60 = np.matmul(H_61,H_652)
        H_71 = np.matmul(H_60,H_761)
        H_72 = np.matmul(H_71,H_762)
        H_70 = np.matmul(H_72,H_763)
        H_80 = np.matmul(H_70,H_87)


        jointPositions[0,:] = H_10[0:3,-1]
        jointPositions[1,:] = H_20[0:3,-1]
        jointPositions[2,:] = H_30[0:3,-1]
        jointPositions[3,:] = H_40[0:3,-1]
        jointPositions[4,:] = H_50[0:3,-1]
        jointPositions[5,:] = H_60[0:3,-1]
        jointPositions[6,:] = H_70[0:3,-1]
        #print('jp', jointPositions, jointPositions.shape)
        #print('h_20', H_10[0:3,-1], H_20[0:3,-1].shape)
        #print('jp1', jointPositions[0,:], jointPositions[0,:].shape)
        H_00 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0]])

        T0e = H_80
        #print(jointPositions)


        H = np.zeros((9, 4, 4))
        H[0] = H_00
        H[1] = H_10
        H[2] = H_20
        H[3] = H_30
        H[4] = H_40
        H[5] = H_50
        H[6] = H_60
        H[7] = H_70
        H[8] = H_80


        return jointPositions, T0e



if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #q = np.array([pi/2,pi/2,pi,-pi/2,0,pi/2,pi/4])

    q = np.array([-0.4,0,0,-2,0,2,pi/4])
    joint_positions, T0e = fk.forward(q)
    #print('H_10', H_10, 'H_10.shape', H_10.shape)
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)

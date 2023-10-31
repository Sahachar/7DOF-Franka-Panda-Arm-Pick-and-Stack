import numpy as np
from lib.calculateFK import FK
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

    def DHm(self, a,al,d,th):
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
        H_10 = self.DHm(0,0,self.l0,0)
        H_21 = self.DHm(0,-pi/2,self.l1,q[0])
        H_321 = self.DHm(0,pi/2,0,q[1])
        H_322 = self.DHm(0,0,self.l2,0)
        H_43 = self.DHm(self.l5,pi/2,self.l3,q[2])
        H_541 = self.DHm(-self.l4,-pi/2,0,q[3])
        H_542 = self.DHm(0,0,self.l6,0)
        H_651 = self.DHm(0,pi/2,self.l7,q[4])
        H_652 = self.DHm(0,0,-self.l11,q[5])
        H_761 = self.DHm(0,0,self.l11,0)
        H_762 = self.DHm(self.l8,pi/2,0,0)
        H_763 = self.DHm(0,0,self.l9,0)
        H_87 = self.DHm(0,0,self.l10,q[6]-pi/4)

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

        H_00 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        T0e = H_80
        #print(jointPositions)
        # Your code ends here

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


        return H, H_10, H_20, H_30, H_40, H_50, H_60, H_70, H_80, jointPositions, T0e


def calcJacobian(q):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q: 0 x 7 configuration vector (of joint angles) [q0,q1,q2,q3,q4,q5,q6]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))
    H, H_10, H_20, H_30, H_40, H_50, H_60, H_70, H_80, joint_positions, T0e = FK().forward(q)
    #print('joint positions', joint_positions)


    z = np.zeros((7, 3))

    #for joint_num in range(0,7):
    #z[0] = getCol(H[0], 2)
    #z[1] = getCol(H[1], 1)
    #z[2] = getCol(-1*H[2], 1)
    #z[3] = getCol(-1*H[3], 0)
    #z[4] = getCol(-1*H[4], 1)
    #z[5] = getCol(-1*H[5], 0)
    #z[6] = getCol(-1*H[6], 1)

    z[0] = getCol(H[1], 2)
    z[1] = getCol(H[2], 2)
    z[2] = getCol(H[3], 2)
    z[3] = getCol(H[4], 2)
    z[4] = getCol(H[5], 2)
    z[5] = getCol(H[6], 2)
    z[6] = getCol(H[7], 2)



    d = np.zeros((9, 3))

    for joint_num in range(0,9):
        d[joint_num] = getCol(H[joint_num], 3)

    rot = np.zeros((7, 3))

    for joint_num in range(0,7):
        rot[joint_num] = crossProd(z[joint_num], (d[8] - d[joint_num+1]))

    r_num = 0

    for c_num in range(0,7):
        J[r_num:r_num+len(rot[c_num]), c_num]       = rot[c_num]
        J[r_num+3:r_num+3+len(rot[c_num]), c_num]   = z[c_num]



    ## STUDENT CODE GOES HERE

    return J

def getCol(homogMatrix, colNum):
    column = homogMatrix[0:-1,colNum]
    return column #3x1

def crossProd(v_1, v_2):
    crossResult = np.cross(v_1, v_2)
    return crossResult


if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q= np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0, np.pi/4])
    fk = FK()
    H, H_10, H_20, H_30, H_40, H_50, H_60, H_70, H_80, joint_positions, T0e = fk.forward(q)
    print(joint_positions, T0e)
    print(np.round(calcJacobian(q),3))

import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.IK_velocity import IK_velocity

class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=500, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        displacement = np.zeros(3)
        axis = np.zeros(3)

        ##Find displacement
        displacement = (target[0:3,3]-current[0:3,3]).reshape(-1,1)
        #print('displacement.shape', displacement.shape)
        ##Find Axis

        R_tin0 = target[0:3,0:3]
        #print('R_tin0', R_tin0)
        R_cin0 = current[0:3,0:3]
        #print('R_cin0', R_cin0)
        R_tinc = R_cin0.T @ R_tin0 #not *
        #print('R_tinc', R_tinc)
        S = (1/2)*(R_tinc - np.transpose(R_tinc))
        #print('S', S)
        a_inc = np.array([[S[2,1]],
                          [S[0,2]],
                          [S[1,0]]])

        a_in0 = R_cin0 @ a_inc
        axis = a_in0
        #print('a_inc', a_inc)
        #print('axis', axis.shape, axis)

        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H


        """

        distance = 0
        angle = 0

        ##Find distance
        distance = np.sqrt((G[0,3]-H[0,3])**2 +
                           (G[1,3]-H[1,3])**2 +
                           (G[2,3]-H[2,3])**2)

        ##Get relative rotation matrix
        R_Gin0 = G[0:3,0:3]
        R_Hin0 = H[0:3,0:3]

        R_GinH = np.matmul(np.linalg.inv(R_Hin0),R_Gin0)

        ##Define inner function of arrcos
        inner_function = (1/2)*(np.trace(R_GinH)-1)

        ##Constrain input to arccos to range [-1,1]
        if inner_function > 1:
            inner_function = 1
        if inner_function < -1:
            inner_function = -1

        ##Find angle
        angle = abs(np.arccos(inner_function))

        return distance, angle

    def is_valid_solution(self,q,target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """
        
        success = False

        ##find actual end effector pose based on candidate solution (joint angles)
        joint_positions, H = IK.fk.forward(q)
        linear_diff, angular_diff = IK.distance_and_angle(H, target)

        if(((self.lower < q).all() and (q < self.upper).all()) and (linear_diff < self.linear_tol) and (angular_diff < self.angular_tol)):
            success = True


        return success

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """
        
        dq = np.zeros(7)

        ##find current frame end effector pose, T0e
        #print('q', q, q.shape)
        joints, T0e = IK.fk.forward(q)

        ##based on #1, lin_vel = displacement, ang_vel = axis
        ##the only way to use displacement_and_axis properly seems to be if I put "self" in front of the function...
        ##but then self would need to be added to this function...
        ##since @staticmethod was written before displacement_and_axis, I can use it without passing "self"
        ##and instead use IK.displacement_and_axis directly
        displacement, axis = IK.displacement_and_axis(target, T0e)
        dq = IK_velocity(q, displacement, axis)

        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1):
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """


        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []

        counter = 0
        while True:

            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q,target)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            # Task Prioritization
            dq = np.zeros(7) # TODO: implement me!
            J = calcJacobian(q)
            null_J = null_space(J)


            dq = dq_ik + np.dot(dq_center , null_J) * null_J / np.linalg.norm(null_J)
            dq = dq.reshape(1,7)
            #print('dq', dq, dq.shape)
            counter = counter + 1
            # Termination Conditions
            if((counter >= self.max_steps) or (np.linalg.norm(dq) < self.min_step_size)): # TODO: check termination conditions
                break # exit the while loop if conditions are met!

            q = q + dq
            q = q.reshape(7,)
        success = self.is_valid_solution(q,target)
        print(success)
        return q, success, rollout

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    target = np.array([
        [0,-1,0,0.3],
        [-1,0,0,0],
        [0,0,-1,.5],
        [0,0,0, 1],
    ])

    target = np.array([
        [1,0,0,0.3],
        [0,1,0,0],
        [0,0,1,.5],
        [0,0,0, 1],
    ])

    q, success, rollout = ik.inverse(target, seed)

    for i, q in enumerate(rollout):
        joints, pose = ik.fk.forward(q)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q, 'd',d,'ang',ang)#, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))


    print("Success: ",success)
    print("Solution: ",q)
    print("Iterations:", len(rollout))

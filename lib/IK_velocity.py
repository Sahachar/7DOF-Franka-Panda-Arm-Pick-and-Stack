import numpy as np
from lib.calcJacobian import calcJacobian
from numpy.linalg import inv

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q: 0 x 7 vector corresponding to the robot's current configuration.
    :param v: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 0 x 7 vector corresponding to the joint velocities. If v and omega
         are infeasible, then dq should minimize the least squares error. If v
         and omega have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros(7)
    
    #print('v_in', v_in.shape, v_in)
    #print('q_in', q_in.shape, q_in)
    #print('omega_in', omega_in.shape, omega_in)
    
    input_vector = np.array(np.concatenate((v_in, omega_in), axis = 0 ))
    #print('input_vector', input_vector, input_vector.shape)
    nan_boolean_array = np.isnan(input_vector)
    nan_boolean_array = nan_boolean_array.reshape(6)
    input_new = input_vector[~nan_boolean_array]

			
    J = calcJacobian(q_in)
    #print('J', J.shape, J)
    J_new = J[~nan_boolean_array,:]
    dq, residuals, rank, s = np.linalg.lstsq(J_new, input_new, rcond=None)


    #print('dq', dq)
    return dq

q_in= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
v_in = np.array([0, 1, 0])
omega_in = np.array([np.nan, 0.1, np.nan])
dq  = IK_velocity(q_in, v_in, omega_in)

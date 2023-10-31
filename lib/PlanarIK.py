import numpy as np
from math import pi, atan2
from numpy import cos, sin, arccos


class PlanarIK:
    """
    Solves the planar IK problem for panda robot arm
    """
    offset = 0.0825  # off axis offset
    offset_5 = 0.088
    l0 = 0.333  # length of link from j0 to j1
    l1 = 0.316  # length of link from j1 to j3
    l3 = 0.384  # length of link from j1 to j5
    l5 = 0.107 + 0.11 # length of link from j5 to gripper point

    # angle offsets due to non straight link
    j1_offset = np.arctan(-offset / l1)
    j3_offset = j1_offset + np.arctan(-offset / l3)
    j5_offset = -j3_offset + np.arctan(-offset_5 / l5) + j1_offset

    def panda_ik(self, physical_target):
        """
        Solves planar IK problem given physical target in x,z plane
        Args:
            physical_target: dictionary containing:
                'o': numpy array of target relative to robot base[x,z]
                'theta': scalar of angle about y axis for target in radians

        Returns:
             q = 2x7 numpy array of joints in radians
        """
        abstract_target = self.physical_to_abstract_target(physical_target)
        abstract_joints = self.rrr_abstract_ik(abstract_target)
        physical_joints = self.physical_to_abstract_joints(abstract_joints)
        return self.convert_to_range(physical_joints)

    def physical_to_abstract_target(self, physical_target):
        """
        Converts physical target to abstract target relative to joint 1 in rrr bot configuration in the xy plane.
        0 configuration in abstract space has arm fully extended pointing along the x axis
        Args:
            physical_target: dictionary containing:
                'o': numpy array of target relative to robot base[x,z]
                'theta': scalar of angle about y axis for target

        Returns:
            abstract_target: dictionary containing:
                'o': numpy array of target relative to j1[x,y]
                'theta': scalar of angle about abstract z axis in radians

        """
        abstract_target = physical_target
        abstract_target['o'] = physical_target['o'] - np.array([0, self.l0])  # shift by length l0
        # Change from positive clockwise to positive counter clockwise and shift angle by joint offsets
        abstract_target['theta'] = -physical_target['theta'] - (-self.j1_offset + self.j3_offset + self.j5_offset)
        return abstract_target

    def physical_to_abstract_joints(self, abstract_joints):
        """
        Converts abstract joints to physical joints
        0 configuration in abstract space has arm fully extended pointing along the x axis
        Args:
            abstract_joints: 2x3 np.array in radians

        Returns:
             physical_joints: 2x7 np.array in radians
        """
        # Zero position is abstract arm fully extended along x axis
        zeros = np.tile(np.array([0, pi / 2 + self.j1_offset, 0, self.j3_offset, 0, pi + self.j5_offset, pi / 4]), (2,1))
        # Add abstract joints accounting for different directions
        physical_joints = zeros + np.array([[0, -abstract_joints[0, 0], 0, abstract_joints[0, 1], 0, abstract_joints[0, 2], 0],
                                            [0, -abstract_joints[1, 0], 0, abstract_joints[1, 1], 0, abstract_joints[1, 2], 0]])
        return physical_joints

    def convert_to_range(self, physical_joints):
        """
        Ensures joints are between -pi and pi
        """
        physical_joints = physical_joints % (2 * pi)
        physical_joints[physical_joints > pi] = physical_joints[physical_joints > pi] - 2 * pi
        return physical_joints

    def rrr_abstract_ik(self, target):
        """
        Solves planar (xy) IK problem for RRR robot where target is relative to 1st joint and zere config of arm has arm
        fully extended along the x axis. Should return both possible IK solutions
        Args:
            target: dictionary containing:
                'o': numpy array of target relative to j1[x,y]
                'theta': scalar of angle about abstract z axis in radians (positive counter clockwise)

        Returns:
            q = 2x3 numpy array of the 3 joints of the RRR robot
        """
        a1 = np.sqrt(self.l1**2.0 + self.offset**2.0)
        a2 = np.sqrt(self.l3**2.0 + self.offset**2.0)
        a3 = np.sqrt(self.l5 ** 2.0 + self.offset_5 ** 2.0)

        print('a1', a1)
        print('a2', a2)
        print('a3', a3)
        q1_a = 0
        q2_a = 0
        q3_a = 0
        q1_b = 0
        q2_b = 0
        q3_b = 0
        # **** Student code goes here ****
        
        target_angles = list(target.values())[0]
        x = target_angles[0]
        y = target_angles[1]
        scalar_angles = list(target.values())[1]
        phi = scalar_angles
        x_prime = x - a3*cos(phi)
        y_prime = y - a3*sin(phi)
        

        q1_a, q2_a, q3_a = system_equations(x_prime, y_prime, phi, 1, a1, a2, a3)
        q1_b, q2_b, q3_b = system_equations(x_prime, y_prime, phi, -1, a1, a2, a3)

        return np.array([[q1_a, q2_a, q3_a], [q1_b, q2_b, q3_b]])

def system_equations(x_prime, y_prime, phi, sigma, a1, a2, a3):
    
        
        num = -1 * (x_prime ** 2 + y_prime ** 2 + a1 ** 2 - a2 ** 2)
        denom = 2 * a1 * np.sqrt(x_prime ** 2 + y_prime ** 2)
        inner_function = num/denom
        
        first_q1   = -y_prime / (np.sqrt(x_prime ** 2 + y_prime ** 2))
        second_q1  = -x_prime / (np.sqrt(x_prime ** 2 + y_prime ** 2))
        
        gamma = atan2(first_q1, second_q1)
        
        q1 = gamma + sigma * arccos(inner_function)
        #***************************************
        
        first_q2  = (y_prime - a1 * sin(q1)) / a2
        second_q2 = (x_prime - a1 * cos(q1)) / a2
        
        q2 = atan2(first_q2, second_q2) - q1
        
        #***************************************
        
        q3 = phi - q1 - q2
        return q1, q2, q3


target_array = np.array([0.5,0.5])
scalar_angle = pi/2

dictionary = {"o": target_array,
              "theta": scalar_angle}
test = list(dictionary.values())[0]
first_value = test[1]
ik = PlanarIK()
final_joint_angles =ik.rrr_abstract_ik(dictionary)


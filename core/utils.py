import rospy

def time_in_seconds():
    """
    :return: the current ROS clock time (not necessarily the same as Wall time i.e. real time)
    :rtype: float
    """
    time_now = rospy.Time.now()
    return time_now.secs + time_now.nsecs * 1e-9

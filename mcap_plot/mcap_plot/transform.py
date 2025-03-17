import math 
import numpy as np

def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = np.empty((4, ))
    q[3] = cr * cp * cy + sr * sp * sy
    q[0] = sr * cp * cy - cr * sp * sy
    q[1] = cr * sp * cy + sr * cp * sy
    q[2] = cr * cp * sy - sr * sp * cy
    return q

def euler_from_quaternion(q):
    t0 = +2.0 * (q[3] * q[0] + q[1] * q[2])
    t1 = +1.0 - 2.0 * (q[0] * q[0] + q[1]*q[1])
    rpy = np.empty((3, ))
    rpy[0] = math.atan2(t0,t1)
    t2 = +2.0 * (q[3] * q[1] - q[2] * q[0])
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    rpy[1] = math.asin(t2)

    t3 = +2.0 * (q[3] * q[2] + q[0] * q[1])
    t4 = +1.0 - 2.0 * (q[1]*q[1] + q[2] * q[2])
    rpy[2] = math.atan2(t3, t4)
    return rpy

def quaternion_mult(q0, q1):
    return [q0[3]*q1[0]+q0[0]*q1[3]+q0[1]*q1[2]-q0[2]*q1[1],
            q0[3]*q1[1]+q0[1]*q1[3]+q0[2]*q1[0]-q0[0]*q1[2],
            q0[3]*q1[2]+q0[2]*q1[3]+q0[0]*q1[1]-q0[1]*q1[0],
            q0[3]*q1[3]-q0[0]*q1[0]-q0[1]*q1[1]-q0[2]*q1[2]]

def rotate_vec(v, q):
    r = np.append(v, 0.0)
    print(np.shape(r))
    q_conj = [-q[0],-q[1],-q[2], q[3]]
    return quaternion_mult(quaternion_mult(q,r),q_conj)[0:3]

NED_ENU_Q = quaternion_from_euler(math.pi, 0., math.pi/2)
ENU_NED_Q = NED_ENU_Q
AIRCRAFT_BASELINK_Q = quaternion_from_euler(math.pi, 0, 0.)

def NED2ENU_vec(v):
    return [v[1], v[0], -v[2]]

def NED2ENU_quat(q):
    return quaternion_mult(NED_ENU_Q, q)

def ENU2NED_vec(v):
    return [v[1], v[0], -v[2]]

def ENU2NED_quat(q):
    return quaternion_mult(ENU_NED_Q, q)

def wxyz2xyzw(q):
    #return q[1:4]+[q[0]]
    return [q[1],q[2],q[3],q[0]]

def xyzw2wxyz(q):
    return [q[3],q[0],q[1],q[2]]
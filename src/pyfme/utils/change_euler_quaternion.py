# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:34:43 2016

@author: Juan
Modified by Jean (jeandb@stanford.edu) on Nov 15th 2018 
"""

import numpy as np
import quaternion
arctan2 = np.arctan2
arcsin = np.arcsin 
cos = np.cos 
sin = np.sin

import pdb

def quatern2euler(Q):
    '''Given a quaternion, the euler_angles vector is returned.

    Parameters
    ----------
    quaternion : array_like
        Nx4 vector with the four elements of the quaternion:
        [q_0, q_x, q_y, q_z]

    Returns
    -------
    euler_angles : array_like
        Nx3 array with the euler angles: [theta, phi, psi]    (rad)

    References
    ----------
       [1] "Quaternions (Com S 477/577 Notes)" Yan-Bin Jia
       [2] "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors" Chapter 5.6 James Diebel, Stanford University
    '''

    try:
        check_unitnorm(Q.T)
    except ValueError:
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        check_unitnorm(Q.T)

    phi = np.arctan2(Q[:,0]*Q[:,1] + Q[:,3]*Q[:,2],
                         1/2 - (Q[:,1]**2 + Q[:,2]**2))
    theta = - np.arcsin(2 * (Q[:,1] * Q[:,3] - Q[:,0] * Q[:,2]))
    psi = np.arctan2(Q[:,3] * Q[:,0] + Q[:,1] * Q[:,2],
                     1/2 - (Q[:,2]**2 + Q[:,3]**2))

    euler_angles = np.array([phi, theta, psi]).T

    return euler_angles


def euler2quatern(euler_angles):
    '''Given the euler_angles vector, the quaternion vector is returned.

    Parameters
    ----------
    euler_angles : array_like
        1x3 array with the euler angles: [phi, theta, psi]    (rad)

    Returns
    -------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_x, q_y, q_z]

    References
    ----------
       [1] "Quaternions (Com S 477/577 Notes)" Yan-Bin Jia
       [2] "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors" Chapter 5.6 James Diebel, Stanford University
    '''

    phi, theta, psi = euler_angles.T
    c_phi = np.cos(phi/2)
    s_phi = np.sin(phi/2)
    c_theta = np.cos(theta/2)
    s_theta = np.sin(theta/2)
    c_psi = np.cos(psi/2)
    s_psi = np.sin(psi/2)

    Q = np.zeros([len(phi) ,4])
    Q[:,0] = c_phi*c_theta*c_psi + s_phi*s_theta*s_psi
    Q[:,1] =-c_phi*s_theta*s_psi + s_phi*c_theta*c_psi
    Q[:,2] = c_phi*s_theta*c_psi + s_phi*c_theta*s_psi
    Q[:,3] = c_phi*c_theta*s_psi - s_phi*s_theta*c_psi

    return Q


def rotate_vector(vector, quat_array):
    '''Given an array of quaternions, rotate every vector in vector.

    Parameters
    ----------
    vector : array_like
        Nx3 array 
    quat_array : array of quaternions as defined by np.quaternion 

    Returns
    -------
    vector : array_like

    TODO
    ----------
    Replace this by v' = v + 2 * r x (s * v + r x v) / m
    where x represents the cross product, s and r are the scalar and
    vector parts of the quaternion, respectively, and m is the sum of
    the squares of the components of the quaternion. Implemented with numba.
   '''
    N  = vector.shape[0]
    V = quaternion.from_float_array(np.hstack((np.zeros((N,1)), vector)))
    return quaternion.as_float_array(
        quat_array * V * np.invert(quat_array))[:,1:] # norm is the square of the II-norm in np.quaternion


def change_basis(vector, quat_array):
    '''Apply the rotation described in quat_array to the basis in which vectors in vector are described.
    A change of basis is about finding the components of the old basis vectors in the new basis, which means applying 
    the inverse rotation.

    Parameters
    ----------
    vector : array_like, coordinates in basis1
        Nx3 array 
    quat_array : array of quaternions as defined by np.quaternion. They describe the rotation from basis1 
    to basis2, i.e. for any corresponding basis vectors from basis1 and basis2, say e1 and e2, we have: e2 = rotate_vector(e1, quat)
    Returns
    -------
    vector : array_like
    '''

    return rotate_vector(vector, np.invert(quat_array))



def check_unitnorm(quaternion):
    '''Given a quaternion, it checks the modulus (it must be unit). If it is
    not unit, it raises an error.

    Parameters
    ----------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_x, q_y, q_z]
    Raises
    ------
    ValueError:
        Selected quaternion norm is not unit
    '''
    q_0, q_1, q_2, q_3 = quaternion
    N = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
    check_value = np.isclose([N], [1]) | np.isnan(N)

    if not check_value.all():
        raise ValueError('There is a non-unit norm quaternion in the input array: norm = {}'.format(N))

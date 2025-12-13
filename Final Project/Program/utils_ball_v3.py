import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.animation as animation

@njit(cache=True)
def signedAngle(u = None,v = None,n = None):
    # This function calculates the signed angle between two vectors, "u" and "v",
    # using an optional axis vector "n" to determine the direction of the angle.
    #
    # Parameters:
    #   u: numpy array-like, shape (3,), the first vector.
    #   v: numpy array-like, shape (3,), the second vector.
    #   n: numpy array-like, shape (3,), the axis vector that defines the plane
    #      in which the angle is measured. It determines the sign of the angle.
    #
    # Returns:
    #   angle: float, the signed angle (in radians) from vector "u" to vector "v".
    #          The angle is positive if the rotation from "u" to "v" follows
    #          the right-hand rule with respect to the axis "n", and negative otherwise.
    #
    # The function works by:
    # 1. Computing the cross product "w" of "u" and "v" to find the vector orthogonal
    #    to both "u" and "v".
    # 2. Calculating the angle between "u" and "v" using the arctan2 function, which
    #    returns the angle based on the norm of "w" (magnitude of the cross product)
    #    and the dot product of "u" and "v".
    # 3. Using the dot product of "n" and "w" to determine the sign of the angle.
    #    If this dot product is negative, the angle is adjusted to be negative.
    #
    # Example:
    #   signedAngle(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    #   This would return a positive angle (π/2 radians), as the rotation
    #   from the x-axis to the y-axis is counterclockwise when viewed along the z-axis.
    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle

    return angle

@njit(cache=True)
def mmt(matrix):
    return matrix + matrix.T

@njit(cache=True)
def getTheta(x0, x1 = None, x2 = None, x3 = None):

    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
      x1 = x0[3:6]
      x2 = x0[6:9]
      x3 = x0[9:12]
      x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0

    n0 = np.cross(m_e0, m_e1)
    n1 = np.cross(m_e2, m_e0)

    # Calculate the signed angle using the provided function
    theta = signedAngle(n0, n1, m_e0)

    return theta

# In the original code, there are probaly TWO sign errors in the expressions for m_h3 and m_h4.
# [Original code: % https://github.com/shift09/plates-shells/blob/master/src/bending.cpp]
# I indicated those two corrections by writing the word "CORRECTION" next
# to them.

@njit(cache=True)
def gradTheta(x0, x1 = None, x2 = None, x3 = None):

    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
      x1 = x0[3:6]
      x2 = x0[6:9]
      x3 = x0[9:12]
      x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 = np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 = np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 = np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 = np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1 = np.cross(m_e0, m_e3)
    m_nn1 = m_nn1 / np.linalg.norm(m_nn1)
    m_nn2 = -np.cross(m_e0, m_e4)
    m_nn2 = m_nn2 / np.linalg.norm(m_nn2)

    m_h1 = np.linalg.norm(m_e0) * m_sinA1
    m_h2 = np.linalg.norm(m_e0) * m_sinA2
    m_h3 = -np.linalg.norm(m_e0) * m_sinA3  # CORRECTION
    m_h4 = -np.linalg.norm(m_e0) * m_sinA4  # CORRECTION
    m_h01 = np.linalg.norm(m_e1) * m_sinA1
    m_h02 = np.linalg.norm(m_e2) * m_sinA2

    # Initialize the gradient
    gradTheta = np.zeros(12)

    gradTheta[0:3] = m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4
    gradTheta[3:6] = m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2
    gradTheta[6:9] = -m_nn1 / m_h01
    gradTheta[9:12] = -m_nn2 / m_h02

    return gradTheta

@njit(cache=True)
def hessTheta(x0, x1 = None, x2 = None, x3 = None):

    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
      x1 = x0[3:6]
      x2 = x0[6:9]
      x3 = x0[9:12]
      x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 = np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 = np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 = np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 = np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1 = np.cross(m_e0, m_e3)
    m_nn1 /= np.linalg.norm(m_nn1)
    m_nn2 = -np.cross(m_e0, m_e4)
    m_nn2 /= np.linalg.norm(m_nn2)

    m_h1 = np.linalg.norm(m_e0) * m_sinA1
    m_h2 = np.linalg.norm(m_e0) * m_sinA2
    m_h3 = -np.linalg.norm(m_e0) * m_sinA3
    m_h4 = -np.linalg.norm(m_e0) * m_sinA4
    m_h01 = np.linalg.norm(m_e1) * m_sinA1
    m_h02 = np.linalg.norm(m_e2) * m_sinA2

    # Gradient of Theta (as an intermediate step)
    grad_theta = np.zeros((12, 1))
    grad_theta[0:3] = (m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4).reshape(-1, 1)
    grad_theta[3:6] = (m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2).reshape(-1, 1)
    grad_theta[6:9] = (-m_nn1 / m_h01).reshape(-1, 1)
    grad_theta[9:12] = (-m_nn2 / m_h02).reshape(-1, 1)

    # Intermediate matrices for Hessian
    m_m1 = np.cross(m_nn1, m_e1) / np.linalg.norm(m_e1)
    m_m2 = -np.cross(m_nn2, m_e2) / np.linalg.norm(m_e2)
    m_m3 = -np.cross(m_nn1, m_e3) / np.linalg.norm(m_e3)
    m_m4 = np.cross(m_nn2, m_e4) / np.linalg.norm(m_e4)
    m_m01 = -np.cross(m_nn1, m_e0) / np.linalg.norm(m_e0)
    m_m02 = np.cross(m_nn2, m_e0) / np.linalg.norm(m_e0)

    # Hessian matrix components
    M331 = m_cosA3 / (m_h3 ** 2) * np.outer(m_m3, m_nn1)
    M311 = m_cosA3 / (m_h3 * m_h1) * np.outer(m_m1, m_nn1)
    M131 = m_cosA1 / (m_h1 * m_h3) * np.outer(m_m3, m_nn1)
    M3011 = m_cosA3 / (m_h3 * m_h01) * np.outer(m_m01, m_nn1)
    M111 = m_cosA1 / (m_h1 ** 2) * np.outer(m_m1, m_nn1)
    M1011 = m_cosA1 / (m_h1 * m_h01) * np.outer(m_m01, m_nn1)

    M442 = m_cosA4 / (m_h4 ** 2) * np.outer(m_m4, m_nn2)
    M422 = m_cosA4 / (m_h4 * m_h2) * np.outer(m_m2, m_nn2)
    M242 = m_cosA2 / (m_h2 * m_h4) * np.outer(m_m4, m_nn2)
    M4022 = m_cosA4 / (m_h4 * m_h02) * np.outer(m_m02, m_nn2)
    M222 = m_cosA2 / (m_h2 ** 2) * np.outer(m_m2, m_nn2)
    M2022 = m_cosA2 / (m_h2 * m_h02) * np.outer(m_m02, m_nn2)

    B1 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn1, m_m01)
    B2 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn2, m_m02)

    N13 = 1 / (m_h01 * m_h3) * np.outer(m_nn1, m_m3)
    N24 = 1 / (m_h02 * m_h4) * np.outer(m_nn2, m_m4)
    N11 = 1 / (m_h01 * m_h1) * np.outer(m_nn1, m_m1)
    N22 = 1 / (m_h02 * m_h2) * np.outer(m_nn2, m_m2)
    N101 = 1 / (m_h01 ** 2) * np.outer(m_nn1, m_m01)
    N202 = 1 / (m_h02 ** 2) * np.outer(m_nn2, m_m02)

    # Initialize Hessian of Theta
    hess_theta = np.zeros((12, 12))

    hess_theta[0:3, 0:3] = mmt(M331) - B1 + mmt(M442) - B2
    hess_theta[0:3, 3:6] = M311 + M131.T + B1 + M422 + M242.T + B2
    hess_theta[0:3, 6:9] = M3011 - N13
    hess_theta[0:3, 9:12] = M4022 - N24
    hess_theta[3:6, 3:6] = mmt(M111) - B1 + mmt(M222) - B2
    hess_theta[3:6, 6:9] = M1011 - N11
    hess_theta[3:6, 9:12] = M2022 - N22
    hess_theta[6:9, 6:9] = -mmt(N101)
    hess_theta[9:12, 9:12] = -mmt(N202)

    # Make the Hessian symmetric
    hess_theta[3:6, 0:3] = hess_theta[0:3, 3:6].T
    hess_theta[6:9, 0:3] = hess_theta[0:3, 6:9].T
    hess_theta[9:12, 0:3] = hess_theta[0:3, 9:12].T
    hess_theta[6:9, 3:6] = hess_theta[3:6, 6:9].T
    hess_theta[9:12, 3:6] = hess_theta[3:6, 9:12].T

    return hess_theta

@njit(cache=True)
def gradEs_hessEs(node0 = None,node1 = None,l_k = None,EA = None):

# Inputs:
# node0: 1x3 vector - position of the first node
# node1: 1x3 vector - position of the last node

# l_k: reference length (undeformed) of the edge
# EA: scalar - stretching stiffness - Young's modulus times area

# Outputs:
# dF: 6x1  vector - gradient of the stretching energy between node0 and node 1.
# dJ: 6x6 vector - hessian of the stretching energy between node0 and node 1.

    ## Gradient of Es
    edge = node1 - node0

    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen
    epsX = edgeLen / l_k - 1
    dF_unit = EA * tangent * epsX
    dF = np.zeros((6))
    dF[0:3] = - dF_unit
    dF[3:6] = dF_unit

    ## Hessian of Es
    Id3 = np.eye(3)
    M = EA * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * ( np.outer( edge, edge ) ) / edgeLen ** 2)

    dJ = np.zeros((6,6))
    dJ[0:3,0:3] = M
    dJ[3:6,3:6] = M
    dJ[0:3,3:6] = - M
    dJ[3:6,0:3] = - M
    return dF,dJ

@njit(cache=True)
def getEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1.0):
    """
    Compute the bending energy for a shell.

    Returns:
    E (scalar): Bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    E = 0.5 * kb * (theta - theta_bar) ** 2

    return E

@njit(cache=True)
def gradEb_hessEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1.0):
    """
    Compute the gradient and Hessian of the bending energy for a shell.

    Parameters:
    x0 (array): Can either be a 3-element array (single point) or a 12-element array.
    x1, x2, x3 (arrays): Optional, 3-element arrays specifying points.
    theta_bar (float): Reference angle.
    kb (float): Bending stiffness.

    Returns:
    dF (array): Gradient of the bending energy.
    dJ (array): Hessian of the bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = dE/dx = 2 * (theta-thetaBar) * gradTheta
    dF = 0.5 * kb * (2 * (theta - theta_bar) * grad)

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = 0.5 * kb * (2 (theta-thetaBar) d theta/dx)
    # J = dF/dx = 0.5 * kb * [ 2 (d theta / dx) transpose(d theta/dx) +
    #       2 (theta-thetaBar) (d^2 theta/ dx^2 ) ]
    hess = hessTheta(x0, x1, x2, x3)  # Replace with your hessTheta function in Python
    dJ = 0.5 * kb * (2 * np.outer(grad, grad) + 2 * (theta - theta_bar) * hess)

    return dF, dJ

@njit(cache=True)
def gradEb_hessEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1.0):
    """
    Compute the gradient and Hessian of the bending energy for a shell.

    Parameters:
    x0 (array): Can either be a 3-element array (single point) or a 12-element array.
    x1, x2, x3 (arrays): Optional, 3-element arrays specifying points.
    theta_bar (float): Reference angle.
    kb (float): Bending stiffness.

    Returns:
    dF (array): Gradient of the bending energy.
    dJ (array): Hessian of the bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = dE/dx = 2 * (theta-thetaBar) * gradTheta
    dF = 0.5 * kb * (2 * (theta - theta_bar) * grad)

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = 0.5 * kb * (2 (theta-thetaBar) d theta/dx)
    # J = dF/dx = 0.5 * kb * [ 2 (d theta / dx) transpose(d theta/dx) +
    #       2 (theta-thetaBar) (d^2 theta/ dx^2 ) ]
    hess = hessTheta(x0, x1, x2, x3)  # Replace with your hessTheta function in Python
    dJ = 0.5 * kb * (2 * np.outer(grad, grad) + 2 * (theta - theta_bar) * hess)

    return dF, dJ

from numba import njit
import numpy as np

@njit(cache=True)
def objfun(qOld, uOld, freeIndex, dt, tol, massVector, massMatrix,
           ks, refLen, edges,
           kb, thetaBar, hinges,
           Fg, visc, 
           ball_radii, ball_active,
           K_contact, mu_contact, ndof_shell):
    
    qNew = qOld.copy()
    ndof_total = qOld.shape[0]
    nv_shell = ndof_shell // 3

    iter_count = 0
    error = 10.0 * tol

    while error > tol and iter_count < 50:

        # initialize forces and Jacobians
        Fb = np.zeros(ndof_total)
        Fs = np.zeros(ndof_total)
        Fc = np.zeros(ndof_total)

        Jb = np.zeros((ndof_total, ndof_total))
        Js = np.zeros((ndof_total, ndof_total))
        Jc = np.zeros((ndof_total, ndof_total))

        # bending shell
        for kHinge in range(hinges.shape[0]):
            node0 = int(hinges[kHinge, 0])
            node1 = int(hinges[kHinge, 1])
            node2 = int(hinges[kHinge, 2])
            node3 = int(hinges[kHinge, 3])

            i0 = 3 * node0
            i1 = 3 * node1
            i2 = 3 * node2
            i3 = 3 * node3

            x0 = qNew[i0:i0+3]
            x1 = qNew[i1:i1+3]
            x2 = qNew[i2:i2+3]
            x3 = qNew[i3:i3+3]

            # If thetaBar is per-hinge, change to thetaBar[kHinge]
            dF, dJ = gradEb_hessEb_Shell(x0, x1, x2, x3, thetaBar, kb)

            # Assemble forces
            Fb[i0:i0+3] -= dF[0:3]
            Fb[i1:i1+3] -= dF[3:6]
            Fb[i2:i2+3] -= dF[6:9]
            Fb[i3:i3+3] -= dF[9:12]

            # Assemble Jacobian (12x12 block)
            nodes = np.array([node0, node1, node2, node3])
            for a in range(4):
                ia = 3 * nodes[a]
                for b in range(4):
                    ib = 3 * nodes[b]
                    Jb[ia:ia+3, ib:ib+3] -= dJ[3*a:3*a+3, 3*b:3*b+3]

        # shell stretching
        for kEdge in range(edges.shape[0]):
            node0 = int(edges[kEdge, 0])
            node1 = int(edges[kEdge, 1])

            i0 = 3 * node0
            i1 = 3 * node1

            x0 = qNew[i0:i0+3]
            x1 = qNew[i1:i1+3]

            dF, dJ = gradEs_hessEs(x0, x1, refLen[kEdge], ks[kEdge])

            # Forces
            Fs[i0:i0+3] -= dF[0:3]
            Fs[i1:i1+3] -= dF[3:6]

            # Jacobian (6x6 block)
            nodes = np.array([node0, node1])
            for a in range(2):
                ia = 3 * nodes[a]
                for b in range(2):
                    ib = 3 * nodes[b]
                    Js[ia:ia+3, ib:ib+3] -= dJ[3*a:3*a+3, 3*b:3*b+3]

        # contact forces between balls and shell nodes
        num_balls = ball_radii.shape[0]

        for kBall in range(num_balls):
            if not ball_active[kBall]:
                continue

            ball_idx_start = ndof_shell + 3 * kBall
            q_ball = qNew[ball_idx_start:ball_idx_start+3]
            radius = ball_radii[kBall]

            for kNode in range(nv_shell):
                node_idx_start = 3 * kNode
                q_node = qNew[node_idx_start:node_idx_start+3]

                # Cheap XY bounding box to prune far nodes
                if np.abs(q_ball[0] - q_node[0]) > radius:
                    continue
                if np.abs(q_ball[1] - q_node[1]) > radius:
                    continue

                F_pair, J_pair = gradFc_hessFc_BallToNode(
                    q_ball, q_node, radius, K_contact, mu_contact
                )

                if not np.any(F_pair):
                    continue

                # Forces
                Fc[node_idx_start:node_idx_start+3] += F_pair[0:3]   # node
                Fc[ball_idx_start:ball_idx_start+3] += F_pair[3:6]   # ball

                # J: 6x6 block
                # Node-Node
                Jc[node_idx_start:node_idx_start+3, node_idx_start:node_idx_start+3] += J_pair[0:3, 0:3]
                # Node-Ball
                Jc[node_idx_start:node_idx_start+3, ball_idx_start:ball_idx_start+3] += J_pair[0:3, 3:6]
                # Ball-Node
                Jc[ball_idx_start:ball_idx_start+3, node_idx_start:node_idx_start+3] += J_pair[3:6, 0:3]
                # Ball-Ball
                Jc[ball_idx_start:ball_idx_start+3, ball_idx_start:ball_idx_start+3] += J_pair[3:6, 3:6]

        # Gravity just adds to forces
        Forces = Fb + Fs + Fc + Fg

        # Viscous forces: Fv = -visc * (qNew - qOld) / dt
        velocity = (qNew - qOld) / dt
        Fv = -visc * velocity
        Forces += Fv

        # Viscous Jacobian: Jv = -visc/dt * I
        Jv = (-visc / dt) * np.eye(ndof_total)

        # Total force Jacobian
        JForces = Jb + Js + Jc + Jv

        # residual and jacobian
        # f(q) = M/dt * ( (qNew - qOld)/dt - uOld ) - Forces
        term1 = (qNew - qOld) / dt - uOld
        f = (massVector / dt) * term1 - Forces

        # J(q) = M/dt^2 - JForces
        # Use massMatrix exactly as original code
        J = massMatrix.copy()
        J /= (dt * dt)
        J -= JForces

        # ---------- REDUCE TO FREE DOFS ----------
        n_free = freeIndex.shape[0]
        J_free = np.zeros((n_free, n_free))
        f_free = np.zeros(n_free)

        for r in range(n_free):
            gr = freeIndex[r]
            f_free[r] = f[gr]
            for c in range(n_free):
                gc = freeIndex[c]
                J_free[r, c] = J[gr, gc]

        # Solve for Newton correction
        dq_free = np.linalg.solve(J_free, f_free)

        # Update
        for i in range(n_free):
            qNew[freeIndex[i]] -= dq_free[i]

        # Error measure
        error = 0.0
        for i in range(n_free):
            error += np.abs(f_free[i])
        iter_count += 1

        print(iter_count, error)

    uNew = (qNew - qOld) / dt
    return qNew, uNew

# Function to set equal aspect ratio for 3D plots
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

def plotShell(qAll, edges, ctime, ball_radii, ball_active, ndof_shell):
    """
    Updated to handle multiple balls.
    ball_radii: Array of radii for all balls.
    ball_active: Boolean array indicating which balls are active.
    """
    qShell = qAll[:ndof_shell]
    X = qShell[0::3]
    Y = qShell[1::3]
    Z = qShell[2::3]
    
    qBall = qAll[ndof_shell:]

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # color mapping
    z_min = -0.4
    z_max = 0.2
    norm = colors.Normalize(vmin=z_min, vmax=z_max)
    cmap = cm.jet 

    # plot edges
    segments = []
    avg_z_list = []
    
    for edge in edges:
        n0, n1 = edge[0], edge[1]
        p0 = (X[n0], Y[n0], Z[n0])
        p1 = (X[n1], Y[n1], Z[n1])
        segments.append([p0, p1])
        avg_z_list.append((Z[n0] + Z[n1]) / 2.0)
    
    edge_colors = cmap(norm(avg_z_list))
    lc = Line3DCollection(segments, colors=edge_colors, linewidths=1.0)
    ax.add_collection3d(lc)

    # plot nodes
    ax.scatter(X, Y, Z, c=Z, cmap=cmap, norm=norm, s=5, depthshade=False)
    
    # plot balls
    num_balls = len(ball_radii)
    for k in range(num_balls):
        # Only plot if active (or you can remove this check to see 'ghost' balls waiting to spawn)
        if ball_active is not None and not ball_active[k]:
            continue
            
        # Extract index for this ball
        idx = k * 3
        bx, by, bz = qBall[idx], qBall[idx+1], qBall[idx+2]
        r = ball_radii[k]
        
        # Plot
        ax.scatter(bx, by, bz, color='r', marker='o', s=50, label=f'Ball {k}')

    ax.set_title(f't={ctime:.4f}')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())

    set_axes_equal(ax)
    plt.draw()
    plt.savefig('shell_contact%0.4f.pdf' % ctime) 
    plt.close(fig)


def animate_shell(q_history, edges, dt_frame, ball_radii, ball_active, ndof_shell, filename="simulation.gif"):
    """
    Updated to handle multiple balls.
    """
    print(f"Generating animation with {len(q_history)} frames...")
    ndof_shell = int(ndof_shell)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear() 
        
        qAll = q_history[frame_idx]
        ctime = frame_idx * dt_frame
        
        # --- Shell ---
        qShell = qAll[:ndof_shell]
        X = qShell[0::3]
        Y = qShell[1::3]
        Z = qShell[2::3]
        
        qBall = qAll[ndof_shell:] # Contains all ball DOFs
        
        # --- Colors ---
        z_min = -0.4
        z_max = 0.2
        norm = colors.Normalize(vmin=z_min, vmax=z_max)
        cmap = cm.jet

        # --- Edges ---
        segments = []
        avg_z_list = []
        for edge in edges:
            n0, n1 = edge[0], edge[1]
            p0 = (X[n0], Y[n0], Z[n0])
            p1 = (X[n1], Y[n1], Z[n1])
            segments.append([p0, p1])
            avg_z_list.append((Z[n0] + Z[n1]) / 2.0)
            
        edge_colors = cmap(norm(avg_z_list))
        lc = Line3DCollection(segments, colors=edge_colors, linewidths=1.0)
        ax.add_collection3d(lc)

        # Nodes
        ax.scatter(X, Y, Z, c=Z, cmap=cmap, norm=norm, s=5, depthshade=False)
        
        # PLOT BALLS (Loop)
        num_balls = len(ball_radii)
        for k in range(num_balls):
            if ball_active is not None and not ball_active[k]:
                continue
                
            idx = k * 3
            bx, by, bz = qBall[idx], qBall[idx+1], qBall[idx+2]
            
            ax.scatter(bx, by, bz, color='r', marker='o', s=50)

        # -formatting
        ax.set_title(f't={ctime:.4f}')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())
        
        set_axes_equal(ax)

    ani = animation.FuncAnimation(fig, update, frames=len(q_history), interval=50)
    
    if filename.endswith('.gif'):
        ani.save(filename, writer='pillow', fps=20)
    elif filename.endswith('.mp4'):
        ani.save(filename, writer='ffmpeg', fps=20)
    
    plt.close(fig)
    print(f"Animation saved to {filename}")

@njit(cache=True)
def gradFc_hessFc_BallToNode(q_ball, q_node, ball_radius, K_contact, mu_contact):
    r = q_ball - q_node
    dist = np.linalg.norm(r)
    Id3 = np.eye(3)
    Fc = np.zeros(6)
    Jc = np.zeros((6, 6))

    # No contact if center is outside sphere
    if dist >= ball_radius:
        return Fc, Jc

    # Regularize distance so we never divide by zero
    eps = 1e-8
    dist_eff = dist
    if dist_eff < eps:
        dist_eff = eps

    # Force magnitude and direction
    delta = ball_radius - dist_eff      # penetration depth (>= 0)
    n = r / dist_eff                    # unit vector from node to ball

    Fn_mag = K_contact * delta          # normal force magnitude
    Fn = Fn_mag * n                     # force on ball along n

    # Force on node is opposite
    Fc[0:3] = -Fn
    Fc[3:6] = Fn

    # Jacobian (stiffness)
    # H = d(F_ball)/d(q_ball)
    H = -K_contact * np.outer(n, n) - (Fn_mag / dist_eff) * (Id3 - np.outer(n, n))

    # Block structure:
    # [ dF_node/dq_node   dF_node/dq_ball ]
    # [ dF_ball/dq_node   dF_ball/dq_ball ]
    Jc[0:3, 0:3] = H
    Jc[3:6, 3:6] = H
    Jc[0:3, 3:6] = -H
    Jc[3:6, 0:3] = -H

    return Fc, Jc

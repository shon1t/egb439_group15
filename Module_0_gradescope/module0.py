from typing import List,Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time

from open3d.cpu.pybind.core import float64


#################### Refresher ####################
def matrix_creation() -> np.ndarray:
    """
    Create a numpy matrix with the values (1,2) in the first row,
    and the values (3,4) in the second row. The datatype should be float64.

    Returns
    -------
    np.ndarray
        The created array.
    """
    return np.array(([1,2],
                    [3,4]), dtype=float)



def matrix_times_matrix(M:np.ndarray) -> np.ndarray:
    """
    Multiply a matrix by itself. Note that this should be matrix
    multiplication, not element-wise multiplication.

    Parameters
    ----------
    M
        A square numpy array of shape NxN.

    Returns
    -------
    np.ndarray
        The result of M matrix multiplied by itself.
    """

    return M.dot(M)

def matrix_times_vector(M:np.ndarray,V:np.ndarray) -> np.ndarray:
    """
    Multiply a matrix by a vector. Note that this should be matrix
    multiplication, not element-wise multiplication.

    Parameters
    ----------
    M
        A square numpy array of shape NxN.
    V
        A numpy array of shape Nx1 (column vector).

    Returns
    -------
    np.ndarray
        The result of M multiplied by V.
    """

    return M.dot(V)

def invmatrix_times_vec(M:np.ndarray,V:np.ndarray) -> np.ndarray:
    """
    Multiply an inverted matrix by a vector. Note that this should be matrix
    multiplication and inversion, not element-wise multiplication or inversion.

    Parameters
    ----------
    M
        A square numpy array of shape NxN.
    V
        A numpy array of shape Nx1 (column vector).

    Returns
    -------
    np.ndarray
        The result of the inverse of M multiplied by V.
    """

    return np.linalg.inv(M).dot(V)

def solve_linear_equation():
    """
    Solve the set of linear equations:
    4y -3x +7 = 0, and
    2x = 3y + 8

    Do this by using a matrix equation.

    Returns
    -------
    np.ndarray
        A 2x1 column vector [x,y] that satisfies the above equations.
    """
    a = np.array([[4, -3],
                  [2, -3]])

    b = np.array([[-7],
                  [8]])

    return np.linalg.solve(a, b)

def plot_curve():
    """
    Plot a graph of y=cos^2(theta), where theta varies from 0 to 2*pi in 100
    steps. Use the np.linspace() function to compute evenly space theta values.
    """
    data = np.linspace(0, 2*np.pi, 100, endpoint=True, dtype=float)
    plt.plot(data, np.cos(data)**2)
    plt.show()
    pass

def SO2(theta:float) -> np.ndarray:
    """
    Create an SO(2) matrix for a given angle.

    Parameters
    ----------
    theta
        The angle of rotation in radians

    Returns
    -------
    np.ndarray
        The SO(2) matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])


def using_SO2(R_AB:np.ndarray,P_A:np.ndarray) -> np.ndarray:
    """
    Compute the position of point P in frame B.

    Parameters
    ----------
    R_AB
        An SO2 matrix representing the rotation from frame A to frame B.
    P_A
        The position of point P in coordinate frame A.

    Returns
    -------
    np.ndarray
        A 2x1 matrix descibing the point P in frame B.
    """
    print("R_AB: ", R_AB)
    print("P_A: ", P_A)
    print(np.linalg.inv(R_AB).dot(P_A))

    return np.linalg.inv(R_AB).dot(P_A)


def SE2(theta:float,translation:np.ndarray) -> np.ndarray:
    """
    Create an SE(2) matrix for a given angle and translation.

    Parameters
    ----------
    theta
        The angle of rotation in radians

    translation
        A 2x1 matrix describing a translation in a 2D plane.

    Returns
    -------
    np.ndarray
        The SE(2) matrix.
    """

    return np.array([[np.cos(theta), -np.sin(theta), translation[0, 0]],
                     [np.sin(theta), np.cos(theta), translation[1,0]],
                    [0, 0, 1]], dtype=float)

def using_SE2(T_AB:np.ndarray,P_A:np.ndarray) -> np.ndarray:
    """
    Compute the position of point P in frame B.

    Parameters
    ----------
    T_AB
        An SE2 matrix representing the rotation from frame A to frame B.
    P_A
        The position of point P in coordinate frame A.

    Returns
    -------
    np.ndarray
        A 2x1 matrix descibing the point P in frame B.
    """
    T = np.linalg.inv(T_AB).dot(np.vstack((P_A[0], P_A[1], 1)))
    return T[0:2]

def plot_triangle(theta:float,translation:np.ndarray):
    """
    Plot an isosceles triangle indicating a provided position and rotation.
    The triangle has a base length of 0.15m and a height of 0.2m.
    In its unshifted position, the triangle points to the right, with the origin
    located at the tip opposite the base of the triangle.

    An isosceles triangle has two sides that are the same length, with the third
    side being referred to as the base.

    Parameters
    ----------
    theta
        Rotation of the triangle in radians.

    translation
        A 2x1 matrix describing a translation in a 2D plane.
    """
    vertices = np.vstack([[0, 0, 1], [-0.2, 0.075, 1], [-0.2, -0.075, 1]])


    T = SE2(theta, translation)

    Tvertices = np.array([T @ v for v in vertices])

    Tvertices = np.vstack((Tvertices, Tvertices[0]))

    plt.plot(Tvertices[:, 0], Tvertices[:, 1], 'r-', label='Transformed Triangle')

    plt.show()
    pass

#################### Helper Functions ####################
def plot_arena(
    axes:Axes,
    robot_pose:np.ndarray,
    trajectories:List[Tuple[str,np.ndarray]]=[],
    object_positions:List[Tuple[str,np.ndarray]]=[],
    covariance_mats:List[Tuple[str,np.ndarray]]=[]
) -> None:
    """
    Plot all relevant information of the arena in an updating figure.
    This information includes the robot's pose, any trajectories, the positions
    of any objects, any confidence bounds, the arena walls, and any other
    information you may desire.
    Some of these things won't make much sense initially as they will only
    become relevant later in the semester, so don't worry about them until
    they become relevant.

    As there is a lot of things that this function does (some of which is
    unclear until later in the semester), this function will not be tested. You
    will need to self assess whether the plots look right (or come ask a tutor).
    You will be doing a lot of testing in this unit, so any errors should become
    obvious eventually. You are free (and recommended) to alter the function as
    you see fit (including adding/removing parameters from the function call)
    during the semester.
    There is some example code at the bottom of this file. Comment it out to run
    this function

    Parameters
    ----------
    axes
        The axes that the data should be drawn on.
    robot_pose
        A (3,1) or np matrix containing the x,y positions and
        theta heading of the robot.
    trajectories
        A list of tuples where each tuple contains a string and a
        np array. The string is the name of the trajectory, and the array
        is a (2,N) numpy array containing the xy coordinates of each point
        in the trajectory. Default is an empty list.
    object_positions
        A list of tuples where each tuple contains a string and a
        np array. The string is the name of the object, and the array
        is a (2,N) numpy array containing the xy coordinates of the object.
        Default is an empty list.
    covariance_mats
        A list of tuples where each tuple contains a string and a
        np array. The string is the name of the object associated with the
        covariance matrix, and the array is the covariance matrix for the
        object's position. Default is an empty list.

    Notes
    -----
        Assume all matrices will be np.float64 type.

        object_positions matrices are (2,N) instead of (2,1) because it is
        sometimes necessary to plot multiple "sightings" of the same object.
        Each column is a single "sighting"

        The arena walls are static and hence are not provided as a parameter.

    Hints
    -----
        It is simplest to clear the figure and redraw everything.

        axes.plot and axes.scatter are your two main drawing functions.

        Use plt.draw() instead of plt.show() after you have drawn all the lines
        you want.

        The strings in each tuple can be used in the legend.

        plt needs some "down time" where code doesn't run before it will
        actually update an image. You can use plt.pause for this. Either place
        it immediately after you call axes.draw, or replace any time.sleep lines
        in your code with plt.pause.

        Plots will automatically try to size the view to best fit the data.
        As new data gets added, it may resize the view as it sees fit, resulting
        in any points shifting or even getting skewed. Due to this,it may be
        nice to "restrict" the plot's view to be the same size as the arena.
        If something that should be showing on you plot is missing, try disabling
        this restriction and letting the view resize automatically. You may find
        that the lines that weren't showing were simply not in the arena according
        to the data you provided, indicating an error somewhere else in your code.

    """

    pass

if __name__ == "__main__":
    # run unit tests
    import os
    import pytest

    # Change to directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run pytest on the private_tests directory
    pytest.main(["."])


    ### Simple test for the plot_arena function.
    # Verify that the plot updates automatically.
    # robot heading will look weird. Feel free to modify you code to make it look
    # more realistic.
    # q = np.array([-1,0,0],dtype=np.float64).reshape(3,1)
    # ax = plt.axes()
    # for _ in range(1000): #
    #     plot_arena(ax,q,[],[],[])
    #     q += np.array([0.02,*np.random.random((2,))*0.02]).reshape(3,1)

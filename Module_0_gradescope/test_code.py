"""
@author: Jesse Haviland
"""

import unittest
from gradescope_utils.autograder_utils.decorators import (
    weight,
    number,
    visibility,
)
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
from typing import List



class TestQuestion1(unittest.TestCase):
    # All marks here add up to 0.0%

    # -------------- Question 1 Tests --------------- #
    @number("1 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_1_import(self):
        """
        Test that the matrix_creation method can be imported
        """

        try:
            from module0 import matrix_creation  # noqa: F401
        except ImportError:
            self.fail("Could not import matrix_creation from module0.py")

    @number("1 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_1_shape(self):
        """
        Test the matrix_creation method returns data with the correct shape.
        """

        from module0 import matrix_creation as foo
        expected_shape = (2,2)
        expected_datatypes = [np.float64]
        args = ()


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size, should be \
                {expected_shape}, not {res.shape}"
            )

        if res.dtype not in expected_datatypes:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )
        

    # -------------- Question 2 Tests --------------- #
    @number("Q2 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_2_import(self):
        """
        Test that the matrix_times_matrix method can be imported
        """

        try:
            from module0 import matrix_times_matrix  # noqa: F401
        except ImportError:
            self.fail("Could not import matrix_times_matrix from module0.py")

    @number("Q2 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_2_shape(self):
        """
        Test the matrix_times_matrix method returns data with the correct shape.
        """

        from module0 import matrix_times_matrix as foo
        args = (np.eye(2,2,dtype=np.float64),)
        expected_shape = (2,2)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size, should be \
                {expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )
        

    # -------------- Question 3 Tests --------------- #
    @number("Q3 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_3_import(self):
        """
        Test that the matrix_times_vector method can be imported
        """

        try:
            from module0 import matrix_times_vector  # noqa: F401
        except ImportError:
            self.fail("Could not import matrix_times_vector from module0.py")

    @number("Q3 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_3_shape(self):
        """
        Test the matrix_times_vector method returns data with the correct shape.
        """

        from module0 import matrix_times_vector as foo
        args = (
            np.eye(2,2,dtype=np.float64),
            np.eye(2,1,dtype=np.float64)
        )
        expected_shape = (2,1)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be \
                {expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )
        
    # -------------- Question 4 Tests --------------- #
    @number("Q4 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_4_import(self):
        """
        Test that the invmatrix_times_vec method can be imported
        """

        try:
            from module0 import invmatrix_times_vec  # noqa: F401
        except ImportError:
            self.fail("Could not import invmatrix_times_vec from module0.py")

    @number("Q4 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_4_shape(self):
        """
        Test the invmatrix_times_vec method returns data with the correct shape.
        """

        from module0 import invmatrix_times_vec as foo
        args = (
            np.eye(2,2,dtype=np.float64),
            np.eye(2,1,dtype=np.float64)
        )
        expected_shape = (2,1)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be \
                {expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )    
    
    # -------------- Question 5 Tests --------------- #
    @number("Q5 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_5_import(self):
        """
        Test that the solve_linear_equation method can be imported
        """

        try:
            from module0 import solve_linear_equation  # noqa: F401
        except ImportError:
            self.fail("Could not import solve_linear_equation from module0.py")

    @number("Q5 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_5_shape(self):
        """
        Test the solve_linear_equation method returns data with the correct shape.
        """

        from module0 import solve_linear_equation as foo
        args = ()
        expected_shape = (2,1)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be \
                {expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes and len(expected_datatypes) > 0:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )
    
    # -------------- Question 6 Tests --------------- #
    @number("Q6 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_plot_curve_import(self):
        """
        Test that the plot_curve method can be imported
        """

        try:
            from module0 import plot_curve  # noqa: F401
        except ImportError:
            self.fail("Could not import plot_curve from module0.py")

    @number("Q6 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_plot_curve_exists(self):
        """
        Test the plot_curve method produces a plot.
        """

        from module0 import plot_curve as foo
        args = ()

        err_msg = ensure_plot_exists(foo,args)

        if err_msg:
            raise ValueError(err_msg)

    # -------------- Question 7 Tests --------------- #
    @number("Q7 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_7_import(self):
        """
        Test that the SO2 method can be imported
        """

        try:
            from module0 import SO2  # noqa: F401
        except ImportError:
            self.fail("Could not import SO2 from module0.py")

    @number("Q7 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_7_shape(self):
        """
        Test the SO2 method returns data with the correct shape.
        """

        from module0 import SO2 as foo
        args = (0,)
        expected_shape = (2,2)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be \
                {expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes and len(expected_datatypes) > 0:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )

    # -------------- Question 8 Tests --------------- #
    @number("Q8 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_8_import(self):
        """
        Test that the using_SO2 method can be imported
        """

        try:
            from module0 import using_SO2  # noqa: F401
        except ImportError:
            self.fail("Could not import using_SO2 from module0.py")

    @number("Q8 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_8_shape(self):
        """
        Test the using_SO2 method returns data with the correct shape.
        """

        from module0 import using_SO2 as foo
        args = (
            np.eye(2,2,dtype=np.float64),
            np.eye(2,1,dtype=np.float64)
        )
        expected_shape = (2,1)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be "
                +"{expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes and len(expected_datatypes) > 0:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )

    # -------------- Question 9 Tests --------------- #
    @number("Q9 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_9_import(self):
        """
        Test that the SE2 method can be imported
        """

        try:
            from module0 import SE2  # noqa: F401
        except ImportError:
            self.fail("Could not import SE2 from module0.py")

    @number("Q9 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_9_shape(self):
        """
        Test the SE2 method returns data with the correct shape.
        """

        from module0 import SE2 as foo
        args = (0,np.eye(2,1,dtype=np.float64))
        expected_shape = (3,3)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be "
                + f"{expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes and len(expected_datatypes) > 0:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )

    # -------------- Question 10 Tests --------------- #
    @number("Q10 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_10_import(self):
        """
        Test that the using_SE2 method can be imported
        """

        try:
            from module0 import using_SE2  # noqa: F401
        except ImportError:
            self.fail("Could not import using_SE2 from module0.py")

    @number("Q10 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_10_shape(self):
        """
        Test the using_SE2 method returns data with the correct shape.
        """

        from module0 import using_SE2 as foo
        args = (
            np.eye(3,3,dtype=np.float64),
            np.eye(2,1,dtype=np.float64)
        )
        expected_shape = (2,1)
        expected_datatypes = [np.float64]


        res = foo(*args)

        if not res.shape == expected_shape:
            raise ValueError(
                f"final array is the incorrect size. It should be "
                + f"{expected_shape}, not {res.shape} for the provided input."
            )

        if res.dtype not in expected_datatypes and len(expected_datatypes) > 0:
            raise ValueError(
                f"The numpy array type should be any of {expected_datatypes},\
                not {res.dtype}"
            )

    # -------------- Question 11 Tests --------------- #
    @number("Q11 Test 1")
    @weight(0)
    @visibility("visible")
    def test_question_plot_triangle_import(self):
        """
        Test that the plot_triangle method can be imported
        """

        try:
            from module0 import plot_triangle  # noqa: F401
        except ImportError:
            self.fail("Could not import plot_triangle from module0.py")

    @number("Q11 Test 2")
    @weight(0)
    @visibility("visible")
    def test_question_plot_triangle_exists(self):
        """
        Test the plot_triangle method produces a plot.
        """

        from module0 import plot_triangle as foo
        args = (np.pi/2,np.array([1,2],dtype=np.float64).reshape(2,1))

        err_msg = ensure_plot_exists(foo,args)

        if err_msg:
            raise ValueError(err_msg)




def compare(
        answer,
        ref,
        tol:float,
        variable_name:str,
        tests:List[str]=["SIZE","SHAPE","DTYPE","VALUE"],
) -> str:
    """
    Compares a provided answer to a provided reference answer, and generates
    appropriate error messages as necessary.
    """
    if isinstance(tests,str):
        # Guard against user forgetting to put single tests inside of a list.
        tests = [tests]
    tests = [test.upper() for test in tests]

    message = ""

    if isinstance(answer,np.ndarray) ^ isinstance(ref,np.ndarray):
        # If one is an array and the other isn't then doing any further comparisons
        # is difficult, so just return this one error message.
        return f"Submitted answer for variable {variable_name} is of type"\
            f" {type(answer)}, but the reference value is {type(ref)}. Either"\
            f" both must be arrays, or neither should be."

    size_g,shape_g,dtype_g = True,True,True
    if isinstance(answer,np.ndarray):
        if "SIZE" in tests and (answer.size != ref.size):
            size_g = False
            message += f"Incorrect number of elements in variable {variable_name}."+\
                f" Expected {ref.size} but received {answer.size} elements."
        
        if "SHAPE" in tests and answer.shape != ref.shape:
            shape_g = False
            message += f"{variable_name} has an incorrect shape. Expected " +\
                f"{ref.shape} but received {answer.shape} instead."
            
        if "DTYPE" in tests and answer.dtype != ref.dtype:
            dtype_g = False
            message += f"Array \"{variable_name}\" has incorrect datatype."+\
                f" Expected {ref.dtype} but received {answer.dtype}"
        
    # No need to test values are correct if the size and/or shape is incorrect.
    if "VALUE" in tests and size_g and shape_g and not np.allclose(answer,ref,0,tol):
        message += f"Incorrect values in variable {variable_name}."
        
        
    return message   

def ensure_plot_exists(student_func,args):
    '''
    Runs a student plotting function and returns an error message if no plot 
    exists or if no lines are in the plot.
    '''
    plt.close("all")
    student_func(*args)
    err_msg = "No lines found in plot or no plot exists."
    for child in plt.gca().get_children():
        if isinstance(child,Line2D):
            err_msg = ""
            break
    plt.close("all")
    return err_msg



if __name__ == "__main__":
    unittest.main()

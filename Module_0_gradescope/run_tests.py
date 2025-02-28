import pytest
import os

if __name__ == "__main__":
    # Change to directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run pytest on the private_tests directory
    pytest.main(["."])

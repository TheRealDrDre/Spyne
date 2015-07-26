# ================================================================== #
# READ/WRITE
# ================================================================== #
# (Still unfinished) Code for reading and writing persistent models
# ================================================================== #


import neural as n

def write_string(s, file=f):
    """
    A string is written as an integer N (specifying the length)
    and a set of N unicode characters
    """
    pass

def read_string(file=f):
    """Reads a string from an input file"""
    # Reads an integer N from the file, then reads N unicode chars 
    pass


def write_projection(p, file=f):
    f.write("%P:")
    f.read()

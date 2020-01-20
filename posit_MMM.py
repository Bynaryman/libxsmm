import sys
sys.path.append('/home/lledoux/Desktop/PhD/ML/python/PySigmoid')
from PySigmoid import *
import struct

set_posit_env(16,1)

def read_matrices():
    A, B = [], []
    with open("./MatA.raw", "rb") as MatAFile:
        data = MatAFile.read(4)
        while data:
            t = struct.unpack("f", data)[0]
            A.append(t)
            data = MatAFile.read(4)
    with open("./MatB.raw", "rb") as MatBFile:
        data = MatBFile.read(4)
        while data:
            t = struct.unpack("f", data)[0]
            B.append(t)
            data = MatBFile.read(4)
    return A,B

def F2P(A, B):
    A_posit = [Posit(x) for x in A]
    B_posit = [Posit(x) for x in B]
    return A_posit, B_posit

def main():
    print('Entering Python Zone')
    A,B = read_matrices()
    A_posit, B_posit = F2P(A, B)
    print(A)
    print(A_posit)
    # C_posit = MMM(A_posit, B_posit)
    # C = P2F(C_posit)
    print('Exiting  Python Zone')


if __name__ == '__main__':
    main()

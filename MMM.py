import sys
sys.path.append('/home/lledoux/Desktop/PhD/ML/python/PySigmoid/PySigmoid')
from PySigmoid import *
import random  # randint
from functools import reduce  # reduce
import operator
import numpy as np
from numpy import linalg as LA  # for the matrices norm

# matrices dimensions
M = 32
N = 32
K = 32

# random values bounds
LOWER_BOUND = -1
UPPER_BOUND = 1

# posit environment
pN = 16
ES = 1
set_posit_env(pN,ES)

'''
    @brief performs a fused dot product of 2 lists of posits(the vectors)
    postpone the rounding at the end
'''
def fused_dot_product(a, b):
    if all(isinstance(x, Posit) for x in (a + b)):
        r = reduce(operator.add, map(lambda x, y: Quire(x) * Quire(y), a,b))
        return Posit(r)
    else:
        raise Exception("Arguments must be lists of posit")

'''
    @brief performs n*m fused dot product to create output matrix
'''
def fused_matmult(a, b):
    transposeB = [list(i) for i in zip(*b)]
    return [[fused_dot_product(row_a, col_b) for col_b in transposeB] for row_a in a]

'''
    @brief performs a non fused dot product (rounding between every mult and accumulation)
'''
def dot_product(a, b):
    if all(isinstance(x, Posit) for x in (a + b)):
        r = reduce(operator.add, map(lambda x, y: x * y, a,b))
        return r
    else:
        raise Exception("Arguments must be lists of posit")
'''
    @brief performs the matmult of posit matrices in row major order
'''
def matmult(a, b):
    transposeB = [list(i) for i in zip(*b)]
    return [[dot_product(row_a, col_b) for col_b in transposeB] for row_a in a]

'''
   @brief extend the sign on 32bits of a >32b signed integer
   @param value(int): integer of 32b that contains a signed value in >32b
   @param bits(int): nb of bits of signed valued in value
'''
def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)

'''
   @brief returns a random posit that will be exactly in a subset of lower and upper bounds
   @returns (tuple)(posit,float) with posit random number and his exact float representation
     c.f. lemma
'''
def get_random_posit( lower_bound, upper_bound ):
    p1 = Posit(lower_bound)
    p2 = Posit(upper_bound)

    # fix bounds if needed
    if float(p1) < lower_bound:
        p1.set_bit_pattern(p1.number+1)
    if float(p2) > upper_bound:
        p2.set_bit_pattern(p2.number-1)

    a = p1.number
    b = p2.number
    a = sign_extend(a,pN)
    b = sign_extend(b,pN)
    x = random.randint(a, b)
    p = Posit()
    strp = '{0:{fill}{width}b}'.format((x + 2**pN) % 2**pN, fill='0', width=pN)
    p.set_bit_pattern(strp)
    f = float(p)
    return p, f

'''
    @brief creates 4 matrices, 2 A & B, 2 in posits, 2 in floats
    @param m (int) dim row mat A
    @param n (int) dim col mat B
    @param k (int) dim col matA, and row mat B
    @returns a 4elem tuple (Ap,Bp,Af,Bf)
       each element of the tuple is a list of list (2d array)
       matrices are stored in row major order
'''
def create_matrices(m, n, k):
    # matrices A
    Ap, Af = [],[]
    for i in range(m):
        rowAip, rowAif = [], []
        for j in range(k):
            p, f = get_random_posit(LOWER_BOUND, UPPER_BOUND)
            rowAip.append(p)
            rowAif.append(f)
        Ap.append(rowAip)
        Af.append(rowAif)

    # matrices B
    Bp, Bf = [],[]
    for i in range(k):
        rowBip, rowBif = [], []
        for j in range(n):
            p, f = get_random_posit(LOWER_BOUND, UPPER_BOUND)
            rowBip.append(p)
            rowBif.append(f)
        Bp.append(rowBip)
        Bf.append(rowBif)

    return Ap, Af, Bp, Bf

def posit_fused_MMM(Ap, Bp):
    return fused_matmult(Ap, Bp)

def posit_MMM(Ap, Bp):
    return matmult(Ap, Bp)

def IEEE_MMM(Af, Bf, ieee_type=np.float64):
    Af = np.array(Af).astype(ieee_type)
    Bf = np.array(Bf).astype(ieee_type)
    return np.matmul(Af, Bf)

def main():
    Ap, Af, Bp, Bf = create_matrices(M, N, K)

    # print(Ap)
    # print(Af)
    # print(Bp)
    # print(Bf)
    Cp        = posit_MMM(Ap, Bp)
    Cp_fused  = posit_fused_MMM(Ap, Bp)
    Cf_double = IEEE_MMM(Af, Bf, np.float64)
    Cf_single = IEEE_MMM(Af, Bf, np.float32)
    Cf_half   = IEEE_MMM(Af, Bf, np.float16)

    # print('posit'+ str(pN) + 'b raw: ', Cp[0][0])
    # print('posit'+ str(pN) + 'b fused: ', Cp_fused[0][0])
    # print('ieee64b', Cf_double[0][0])
    # print('ieee32b', Cf_single[0][0])
    # print('ieee16b', Cf_half[0][0])

    # cast output posit to float64, lemma says that there is exact representation
    Cp_ieee64 = np.array(Cp).astype(float)
    Cp_fused_ieee64 = np.array(Cp_fused).astype(float)

    # default norm is Frobenious, a norm 2
    dist_w_Cp        = LA.norm(np.subtract(Cf_double, Cp_ieee64))
    dist_w_Cp_fused  = LA.norm(np.subtract(Cf_double, Cp_fused_ieee64))
    dist_w_Cf_single = LA.norm(np.subtract(Cf_double, Cf_single))
    dist_w_Cf_half   = LA.norm(np.subtract(Cf_double, Cf_half))

    print('Frobenious Norms of C oracle - C arith i')
    print('distance with posit'+str(pN)+':', dist_w_Cp)
    print('distance with posit fused'+str(pN)+':', dist_w_Cp_fused)
    print('distance with single precision ieee(32b):', dist_w_Cf_single)
    print('distance with half precision ieee(16b):', dist_w_Cf_half)

if __name__ == '__main__':
    main()

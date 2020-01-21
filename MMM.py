import sys
sys.path.append('/home/lledoux/Desktop/PhD/ML/python/PySigmoid/PySigmoid')
from PySigmoid import *
import random  # randint
from functools import reduce  # reduce
import operator
import numpy as np
from numpy import linalg as LA  # for the matrices norm
import matplotlib.pyplot as plt
import seaborn as sns

# random values bounds
LOWER_BOUND = -1
UPPER_BOUND = 1

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
def get_random_posit( lower_bound, upper_bound, pN ):
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
    @param pN (int) bitwidth of posit config
    @returns a 4elem tuple (Ap,Bp,Af,Bf)
       each element of the tuple is a list of list (2d array)
       matrices are stored in row major order
'''
def create_matrices(m, n, k, pN):
    # matrices A
    Ap, Af = [],[]
    for i in range(m):
        rowAip, rowAif = [], []
        for j in range(k):
            p, f = get_random_posit(LOWER_BOUND, UPPER_BOUND, pN)
            rowAip.append(p)
            rowAif.append(f)
        Ap.append(rowAip)
        Af.append(rowAif)

    # matrices B
    Bp, Bf = [],[]
    for i in range(k):
        rowBip, rowBif = [], []
        for j in range(n):
            p, f = get_random_posit(LOWER_BOUND, UPPER_BOUND, pN)
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

def benchmark(posit_width, posit_es, matrix_size):
    # matrices dimensions
    M = matrix_size
    N = matrix_size
    K = matrix_size

    # posit environment
    pN = posit_width
    ES = posit_es
    set_posit_env(pN,ES)
    Ap, Af, Bp, Bf = create_matrices(M, N, K, pN)

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

    return [dist_w_Cp, dist_w_Cp_fused, dist_w_Cf_single, dist_w_Cf_half]

def main():

    posit_configs = [(4,0),(8,0),(16,1),(32,2)]
    matrix_sizes = [1<<x for x in range(1,7)]
    values = []

    for i in matrix_sizes:
        distances = []
        dist_f_single, dist_f_half = 0,0
        for j in posit_configs:
            a,b,c,d = benchmark(j[0],j[1],i)
            distances.append(a)
            distances.append(b)
            dist_f_single = c
            dist_f_half = d
        distances.append(dist_f_half)
        distances.append(dist_f_single)
        values.append(distances)

    transposeValues = [list(i) for i in zip(*values)]
    labels = [
            'posit<4,0>',
            'quire<4,0>',
            'posit<8,0>',
            'quire<8,0>',
            'posit<16,1>',
            'quire<16,1>',
            'posit<32,2>',
            'quire<32,2>',
            'ieee<16,5>',
            'ieee<32,8>'
            ]

    for i,j in enumerate(transposeValues):
        plt.plot(matrix_sizes, transposeValues[i], label=labels[i])
    plt.legend()
    plt.yscale('log')
    plt.xticks(matrix_sizes)
    plt.xlabel('Matrix Size (square)')
    plt.ylabel('Log10(distance)')
    plt.title('Frobenious Distance of MMM between ieee64(oracle) and arithimetic-i')
    plt.show()

if __name__ == '__main__':
    main()

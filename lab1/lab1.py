# item 1
import numpy as np
from math import pow
n = 10

x1 = np.linspace(-1, 1, n)
print(x1)
F=np.array([[x**i for i in range(n)] for x in x1])
print(F)
#g = lambda i, j: x1[i] ** j
# F = np.fromfunction(lambda i, j: pow(x1[i], j), (n, 6))
# print(F)
#def k(i, j):
#    print(x1[i]**j)
#k(1, 5)
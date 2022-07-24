import numpy as np
import matplotlib.pyplot as plt

sigma = 1
n = 500

# генерація за рівномірним розподілом
X = np.linspace(-1, 1, n)

# матриця плану експерименту
F=np.array([[x**i for i in range(6)] for x in X])
print("Matrix F:\n", F)

# вектор похибок
epsilon = np.random.normal(0, 1, n)
# print(epsilon)
# вектор параметрів
teta = np.array([-0.5, 2.0, 1.0, 1.5, 4.2, 1.0])
# print(teta)

# обчислюємо вектор результатів спостережень
Y = np.dot(F, teta) + epsilon
# print(Y)

# транспонуємо матрицю F
FT = np.matrix(F).T
# print(FT)
# множимо транспоновану матрицю F на матрицю F 
FF = np.dot(FT, F)
# знаходимо обернену матрицю
FF = np.linalg.inv(FF)
print("Matrix: \n", FF)
print("Determinant: \n", np.linalg.det(FF))
# обчислюємо МНК оцінку
tetaF = np.dot(np.dot(FF, FT), Y)
print("Least squares values: \n", tetaF)

# обчислюємо дисперсію МНК оцінок
Dtetamatrix = np.dot(sigma, FF)
Dteta = np.diagonal(Dtetamatrix)
print("Dispersion: \n", Dteta)

# матриці тета з шапочкою перетворюємо в масив
tetaF = np.asarray(tetaF).flatten()
# print(tetaF)
def nu(t, x):
    nuteta = 0
    for i in range(6):
        nuteta += t[i] * x**i   
    return nuteta

plt.plot(X, nu(teta, X), label='teta')
plt.plot(X, nu(tetaF, X), label='mnk_teta')
plt.show()

import numpy as np
from scipy.stats import norm, t as student
from math import sqrt

n = 150
m = 6

# генерація за рівномірним розподілом
X = np.linspace(-1, 1, n)  
# print(X)

# матриця плану експерименту
F=np.array([[x**i for i in range(6)] for x in X])
# print("Matrix F:\n", F)

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
FT = F.T
# множимо транспоновану матрицю F на матрицю F 
FF = np.dot(FT, F)
# знаходимо обернену матрицю
FF = np.linalg.inv(FF)
# обчислюємо МНК оцінку
tetaF = np.dot(np.dot(FF, FT), Y)
print("МНК-оцінка: \n", tetaF)

# дані до лабораторної роботи №2
alpha = 0.05
a = -0.4 # teta1 = a 
i = 0

# знаходження alpha/2-квантиля нормального розподілу
normq = norm.ppf(1 - (alpha / 2), 0, 1)
print("Значення квантиля нормального розподілу:", normq)

# знаходження alpha/2-квантиля розподілу Стьюдента
tq = student(n-m).ppf(1 - (alpha / 2))
print("Значення квантиля розподілу Стьюдента:",tq)

# перший випадок - статистика, яка розподілена за нормальним законом
u = (tetaF[i] - a) / sqrt(FF[i][i])
print("Статистика u =", u)

# другий випадок - статистика, яка має розподіл Стьюдента
FtetaF = np.dot(F, tetaF)
s = sqrt(np.dot((Y - FtetaF.T), (Y - FtetaF)) / (n - m))
t = (tetaF[i] - a) / (s * sqrt(FF[i][i]))
print("Статистика t =", t)

# знаходження альфа
if abs(u) > abs(normq):
    print("Гіпотеза відхилена")
else:
    print("Гіпотеза приймається")

if abs(t) > abs(tq):
    print("Гіпотеза відхилена")
else:
    print("Гіпотеза приймається")   

# знаходження найменшого альфа
alphanmin = 0
alphatmin = 0

for i in np.arange(1.0, 0.0, -0.0001):
    tempnormq = norm.ppf(1 - (i / 2), 0, 1)
    temptq = student(n-m).ppf(1 - (i / 2))

    if abs(u) > abs(tempnormq):
        alphanmin = i
    if abs(t) > abs(temptq):
        alphatmin = i
print("Найменше alpha при якому гіпотеза приймається(нормальний розподіл):", 
      alphanmin)
print("Найменше alpha при якому гіпотеза приймається(розподіл Стьюдента):", 
      alphatmin)
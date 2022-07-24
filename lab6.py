import numpy as np
import matplotlib.pyplot as plt

variant = 45 % 5 + 1
print("Номер варіанту:", variant)
if variant == 1:
    print("Логарифмічна регресія\n")

n = 100
m = 6
alpha = 0.05
# вектор параметрів
teta = np.array([-0.5, 2.0, 1.0, 1.5, 4.2, 1.0])

# генерація за рівномірним розподілом
X = np.random.uniform(0.0, 1.0, n)

# моделюємо вектор похибок
epsilon = np.random.normal(0, 1, n)

# матриця плану експерименту
F=np.array([[x**i for i in range(1, 6)] for x in X])
# прологарифмуємо елементи матриці F та додамо 1 в перший стовпик матриці
F = np.insert(np.log(F), 0, 1, axis=1)
# print("Matrix F:\n", F) 
# print(np.full((n), 1))

# обчислюємо вектор результатів спостережень
Y = np.dot(F, teta) + epsilon

# знаходимо МНК-оцінку
# транспонуємо матрицю F
FT = F.T
# множимо транспоновану матрицю F на матрицю F 
FF = np.dot(FT, F) + np.diag([0] + [0.0001 for i in range(5)])
# знаходимо обернену матрицю
FF = np.linalg.inv(FF)
# обчислюємо МНК оцінку
tetaF = np.dot(np.dot(FF, FT), Y)
print("МНК-оцінка:", tetaF, "\n")

# знаходимо Y за МНК-оцінкою
YF = np.dot(F, tetaF)

# обчислюємо множинний коефіцієнт детермінації
R2 = 1 - np.sum((YF - Y) ** 2) / np.sum((np.mean(Y) - Y) ** 2)
print("Множинний коефіцієнт детермінації", R2, "\n")

# будуємо графіки регресійних кривих
order = np.argsort(X)

plt.figure(figsize=(10, 8))
line1 = plt.plot(X[order], Y[order], label='teta')
line2 = plt.plot(X[order], YF[order], label='mnk_teta')
plt.legend()
plt.show()

# оцінимо шанси здати екзамен
scores = np.array([1, 1, 1, 1, 0.01])
# прологарифмуємо та додамо 1
scores = np.insert(np.log(scores), 0, 1)
print("Прологарифмовані оцінки:", scores)
y_scores = np.dot(scores, tetaF)
P = 1/(1+np.exp(-y_scores))
print("Шанси здати екзамен:", P, "\n")



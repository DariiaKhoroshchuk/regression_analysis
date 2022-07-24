import numpy as np
from scipy.stats import chi2, t

n = 100
m = 6
alpha = 0.05

# вектор параметрів
teta = np.array([-0.5, 2.0, 1.0, 1.5, 4.2, 1.0])
# генерація за рівномірним розподілом
X = np.random.uniform(-1.0, 1.0, n)

# моделюємо вектор похибок
epsilon = np.random.normal(0, 1, n)

# матриця плану експерименту
F=np.array([[x**i for i in range(m)] for x in X])
# print("Matrix F:\n", F) 
# обчислюємо вектор результатів спостережень
Y = np.dot(F, teta) + epsilon

# з матриці плану F видаляємо стовбець одиниць
F = np.delete(F, 0, 1)
# з teta видаляємо перший елемент
teta_new = np.delete(teta, 0)

# ПЕРЕВІРКА МУЛЬТИКОЛІНЕАРНОСТІ
# знаходимо кореляційну матрицю
K = np.corrcoef(F.T)

# знаходимо табличне значення chi
chi_tabl = chi2.ppf(1-alpha, 0.5*(m-1)*(m-2))
# обчислюємо розрахункове значення chi
chi_roz = -int(n - 1 - (2 * (m - 1) + 5) / 6) * np.log(np.linalg.det(K))
print('Табличне значення:', chi_tabl)
print('Розраховане значення:', chi_roz, '\n')

while chi_roz > chi_tabl:
    m -= 1
    if m == 2:
        break
    Z = np.linalg.inv(K)
    R = np.array([[-Z[i][j] / np.sqrt(Z[i][i] * Z[j][j]) for i in range(m)]
                  for j in range(m)])
    T = np.array([[R[i][j] * np.sqrt(n - m - 1) / np.sqrt(1 - R[i][j] ** 2)
                   if i != j else 1000 for i in range(m)] for j in range(m)])
    t_tabl = t.ppf(1 - alpha, n -m - 1)
    to_del = -1
    for i in range(m):
        for j in range(i + 1, m):
            if T[i][j] > t_tabl:
                to_del = i
    if to_del == -1:
        break
    F = np.delete(F, to_del, 1)
    teta_new = np.delete(teta_new, to_del)
    K = np.corrcoef(F.T)
    chi_roz = -int(n - 1 - (2 * m + 5) / 6) * np.log(np.linalg.det(K))
    chi_tabl = chi2.ppf(1 - alpha, 0.5 * m * (m - 1))

# АВТОКОРЕЛЯЦІЯ
# ЦИКЛІЧНИЙ КОЕФІЦІЄНТ АВТОКОРЕЛЯЦІЇ
m = len(teta_new)
# транспонуємо матрицю F
FT = F.T
# множимо транспоновану матрицю F на матрицю F 
FF = np.dot(FT, F)
# знаходимо обернену матрицю
FF = np.linalg.inv(FF)
# обчислюємо МНК оцінку
tetaF = np.dot(np.dot(FF, FT), Y)

YF = np.dot(F, tetaF)

U = Y - YF

# обчислюємо циклічний коефіцієнт автокореляції
r = sum([U[i] * U[i - 1] for i in range(1, n)]) / sum(U ** 2)
print("Циклічний коефіцієнт автокореляції r:", r, "\n")

# КРИТЕРІЙ ДАРБІНА-УОТСОНА
d = sum([(U[i] - U[i - 1]) ** 2 for i in range(1, n)]) / sum(U ** 2)

de = 1.63
du = 1.72

if du < d < 4 - du:
    print("Гіпотеза про відсутність автокореляції приймається.\n")
elif de < d < du or 4 - du < d < 4 - de:
    print("Область невизначеності критерію.\n")
elif 0 < d < de:
    print("Приймається альтернативна гіпотеза про позитивну автокореляцію.\n")
elif 4 - de < d < 4:
    print("Приймається альтернативна гіпотеза про негативну автокореляцію.\n")

# МЕТОД ЕЙТКЕНА
ro = n * r / (n - 1) + (len(tetaF) + 1) / n

S = np.array([[ro ** abs(i - j) for i in range(1, n + 1)]
              for j in range(1, n + 1)])

FSF = np.linalg.inv(np.dot(np.dot(FT, np.linalg.inv(S)), F))
tetaFetkey = np.dot((np.dot(np.dot(FSF, FT), np.linalg.inv(S))), Y)
print("МНК-оцінка за методом Ейткена:", tetaFetkey)
print("МНК-оцінка:", tetaF)
print("Таблична МНК-оцінка:", teta_new, "\n")

# ДИСПЕРСІЯ
tetaS = (1 / (n - m)) * np.dot((Y - np.dot(F, tetaF)), (Y - np.dot(F, tetaF)))
tetaSetkey  = (1 / (n - m)) * np.dot((Y - np.dot(F, tetaFetkey)), 
                                     (Y - np.dot(F, tetaFetkey)))
print("Дисперсія для МНК-оцінки:", tetaS)
print("Дисперсія для МНК-оцінки за методом Ейткена:", tetaSetkey)
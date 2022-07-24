import numpy as np
import statistics
from scipy.stats import chi2, f


r = 45
n = 300
m = 6

# вектор параметрів
teta = np.array([-0.5, 2.0, 1.0, 1.5, 4.2, 1.0])

# генерація за рівномірним розподілом
X = np.random.uniform(-1.0, 1.0, n)

# матриця плану експерименту
F=np.array([[x**i for i in range(6)] for x in X])
# print("Matrix F:\n", F) 

# моделюємо вектор похибок для пункту а
epsilonA = np.array([np.random.normal(0, 1 + i / r, 1) 
                     for i in range(1, n + 1)]).reshape(-1)
# моделюємо вектор похибок для пункту б
epsilonB = np.random.normal(0, 1, n)

# обчислюємо вектор результатів спостережень для пункту а
YA = np.dot(F, teta) + epsilonA
# обчислюємо вектор результатів спостережень для пункту б
YB = np.dot(F, teta) + epsilonB

# транспонуємо матрицю F
FT = F.T
# множимо транспоновану матрицю F на матрицю F 
FF = np.dot(FT, F)
# знаходимо обернену матрицю
FF = np.linalg.inv(FF)
# обчислюємо МНК оцінку
tetaFA = np.dot(np.dot(FF, FT), YA)
print("МНК-оцінка для пункту А: \n", tetaFA)
tetaFB = np.dot(np.dot(FF, FT), YB)
print("МНК-оцінка для пункту Б: \n", tetaFB, '\n')

# Перевіряємо модель на наявність гетероскедастичності за критерієм мю
k = 4

# поділ числа на доданки
def divide(n, l):
    pp=[]
    if l >= n:
        print("Error")
    else:
        for i in range(l):
            p = n // l
            pp.append(p)
            n -= p
            l -= 1
    return pp
P = divide(n, k)
# print(P)
# матриця
# рядки матриці підмасиви ігрика
def mm(Y):
    Ym = []
    for i in range(k):
        Yj = []
        for j in range(P[i]):
            Yj.append(Y[j])
        Y = Y[P[i]:]
        Ym.append(Yj)
    return Ym

YAm = YA
YBm = YB

YAmatrix = mm(YAm)
YBmatrix = mm(YBm)
# print(YAm, YBm)
# print(YAmatrix, YBmatrix)
# обчислення суми квадратів відхилень
S_A = [] 
S_B = []       
for i in range(k):
    Sr_A = 0 
    Sr_B = 0
    for j in YAmatrix[i]:
        Sr_A += pow(j - statistics.mean(YAmatrix[i]), 2)
    for l in YBmatrix[i]:
        Sr_B += pow(l - statistics.mean(YBmatrix[i]), 2)
    S_A.append(Sr_A)    
    S_B.append(Sr_B)
S_A_sum = sum(S_A)
S_B_sum = sum(S_B)
# print(S_A)
# print(S_B)
# print(S_A_sum, S_B_sum)
# обчислення критерію мю
mulA = 1
mulB = 1

for s in range(k):
    mulA *= (S_A[s] / P[s]) ** (P[s] / 2)
    mulB *= (S_B[s] / P[s]) ** (P[s] / 2)
muA = -2 * np.log(mulA / ((S_A_sum / n) ** (n / 2))) 
muB = -2 * np.log(mulB / ((S_B_sum / n) ** (n / 2)))

# знаходимо табличне значення критерію мю
chi = chi2.ppf(0.95, k - 1)


print('Критерій мю:')
print('Для пункту А:', muA)
print('Для пункту Б:', muB)
print('Табличне значення:', chi, '\n')

print("Перевірка на гетероскеданстичність за мю критерієм:")
print("Для випадку A:")
if chi > muA:
    print("Явище гетероскеданстичності відсутнє.")
else:
    print("Спостерігається гетероскеданстичність.")
    
print("Для випадку B:")
if chi > muB:
    print("Явище гетероскеданстичності відсутнє.\n")
else:
    print("Спостерігається гетероскеданстичність.\n")
    
# перевіряємо модель на гетероскедастичність 
# за критерієм Гольдфельда-Квандта
order = np.argsort(X)
c = int((4 * n) / 15)

print("Перевірка на гетероскеданстичність за Гольдфельда-Квандта:")
# розділяємо на дві сукупності
# відкидаємо с спостережень, 
# які розміщені в центрі вектора вхідних даних Х

# для першої сукупності
# вектор спостережень
X_new1 = X[order][0:(n-c)//2]

# матриця плану експерименту
Fnew1=np.array([[x**i for i in range(6)] for x in X_new1])

# обчислюємо вектор результатів спостережень для пункту А
YAnew1 = np.dot(Fnew1, teta) + epsilonA[0:(n-c)//2]
# обчислюємо вектор результатів спостережень для пункту Б
YBnew1 = np.dot(Fnew1, teta) + epsilonB[0:(n-c)//2]

# транспонуємо матрицю F
FTnew1 = Fnew1.T
# множимо транспоновану матрицю F на матрицю F 
FFnew1 = np.dot(FTnew1, Fnew1)
# знаходимо обернену матрицю
FFnew1 = np.linalg.inv(FFnew1)
# обчислюємо МНК оцінку
tetaFAnew1 = np.dot(np.dot(FFnew1, FTnew1), YAnew1)
print("МНК-оцінка для першої сукупності для пункту А: \n", tetaFAnew1)
tetaFBnew1 = np.dot(np.dot(FFnew1, FTnew1), YBnew1)
print("МНК-оцінка для першої сукупності для пункту Б: \n", tetaFBnew1, '\n')


# для другої сукупності
# вектор спостережень
X_new2 = X[order][(n+c)//2:]

# матриця плану експерименту
Fnew2=np.array([[x**i for i in range(6)] for x in X_new2])

# обчислюємо вектор результатів спостережень для пункту А
YAnew2 = np.dot(Fnew2, teta) + epsilonA[(n+c)//2:]
# обчислюємо вектор результатів спостережень для пункту Б
YBnew2 = np.dot(Fnew2, teta) + epsilonB[(n+c)//2:]

# транспонуємо матрицю F
FTnew2 = Fnew2.T
# множимо транспоновану матрицю F на матрицю F 
FFnew2 = np.dot(FTnew2, Fnew2)
# знаходимо обернену матрицю
FFnew2 = np.linalg.inv(FFnew2)
# обчислюємо МНК оцінку
tetaFAnew2 = np.dot(np.dot(FFnew2, FTnew2), YAnew2)
print("МНК-оцінка для другої сукупності для пункту А: \n", tetaFAnew2)
tetaFBnew2 = np.dot(np.dot(FFnew2, FTnew2), YBnew2)
print("МНК-оцінка для другої сукупності для пункту Б: \n", tetaFBnew2, '\n')

# знайти суму квадратів залишків S1 та S2 за першою і другою моделями
Sr_A_new1 = sum((YAnew1 - np.dot(Fnew1, tetaFAnew1)) ** 2)
Sr_A_new2 = sum((YAnew2 - np.dot(Fnew2, tetaFAnew2)) ** 2)

Sr_B_new1 = sum((YBnew1 - np.dot(Fnew1, tetaFBnew1)) ** 2)
Sr_B_new2 = sum((YBnew2 - np.dot(Fnew2, tetaFBnew2)) ** 2)

# розраховуємо критерій Фішера
if Sr_A_new2 > Sr_A_new1:
    FisherA = Sr_A_new2 / Sr_A_new1
else:
    FisherA = Sr_A_new1 / Sr_A_new2
 
if Sr_B_new2 > Sr_B_new1:
    FisherB = Sr_B_new2 / Sr_B_new1
else:
    FisherB = Sr_B_new1 / Sr_B_new2

# розраховуємо табличне значення Фішера
Fi = f.ppf(0.95,len(X_new1) - 1, len(X_new2) - 1)


print('Сума квадратів залишків S1')
print('Для пункту А:', Sr_A_new1)
print('Для пункту Б:', Sr_B_new1, '\n')

print('Сума квадратів залишків S2')
print('Для пункту А:', Sr_A_new2)
print('Для пункту Б:', Sr_B_new2, '\n')

print('Критерій Фішера:')
print('Для пункту А:', FisherA)
print('Для пункту Б:', FisherB)
print('Табличне значення:', Fi, '\n')

print("Для випадку A:")
if Fi > FisherA:
    print("Явище гетероскеданстичності відсутнє.")
else:
    print("Спостерігається гетероскеданстичність.")
    
print("Для випадку B:")
if Fi > FisherB:
    print("Явище гетероскеданстичності відсутнє.\n")
else:
    print("Спостерігається гетероскеданстичність.\n")  
    
    
# знайдемо дисперсію
N = 45

YFtetaA = YA - np.dot(F, tetaFA)
R0A =  np.dot(YFtetaA.T, YFtetaA) 
RNrA = np.divide(R0A, (N - m))

YFtetaB = YB - np.dot(F, tetaFB)
R0B =  np.dot(YFtetaB.T, YFtetaB)
RNrB = np.divide(R0B, (N - m))

print('Дисперсія МНК-оцінки для пункту A:', RNrA)
print('Дисперсія МНК-оцінки для пункту Б:', RNrB, '\n')

# знайдемо оцінки параметрів за методом Ейткена
S = np.diag(1 / (1/X))
Sinv = np.linalg.inv(S)
FSF = np.linalg.inv(np.dot(np.dot(FT, Sinv), F))
tetaFAetkey = np.dot((np.dot(np.dot(FSF, FT), Sinv)), YA)
print('МНК-оцінка за методом Ейткена для пункту А: \n', tetaFAetkey)

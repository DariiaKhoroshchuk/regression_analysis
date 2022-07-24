import numpy as np
import scipy.optimize
import scipy.stats

n = 100
m = 6
k = 4

# генерація за рівномірним розподілом
X = np.linspace(-1, 1, n)  
# print(X)

# вектор кратностей
multiplicity = np.random.randint(1, 4, n)

# матриця плану експерименту
F=np.array([[x**i for i in range(6)] for x in X])
print("Матриця F:\n", F, end="\n\n")

# знаходимо кількість x
q = np.sum(multiplicity)

# вектор похибок
epsilon = np.random.normal(0, 1, q)

# усереднений вектор похибок
new_epsilon = []
for i in range(n):
    sum_of_eps = 0
    for j in range(multiplicity[i]):
        sum_of_eps += epsilon[j]
    new_epsilon.append(sum_of_eps / multiplicity[i])
    epsilon = epsilon[multiplicity[i]:] 
# print(new_epsilon)

# вектор параметрів
teta = np.array([-0.5, 2.0, 1.0, 1.5, 4.2, 1.0])

# обчислюємо вектор результатів спостережень
Y = np.dot(F, teta) + new_epsilon
# print(Y)

# задаємо матрицю W
W = np.diag(1 / multiplicity)
# print(W)

# задаємо матрицю L
N = 45
L = np.zeros((k, n))
for i in range(k):
    for j in range(n):
        L[i][j] = (100 + N) * N * np.random.uniform(0, 1)
print("Матриця L:\n", L, end="\n\n")

# обчислення матриці T
T = np.dot(L, F)
print("Матриця T:\n", T, end="\n\n")

print("Вектор параметрів:", teta)

# знаходимо МНК оцінку
r = np.linalg.matrix_rank(F)
Winv = np.linalg.inv(W)
FT = F.T
if r == m:
    tetaF1 = np.linalg.inv(np.dot(np.dot(FT, Winv), F))
    tetaF = np.dot(np.dot(np.dot(tetaF1, FT), Winv), Y)
else:
    tetaF1 = np.linalg.pinv(np.dot(np.dot(FT, Winv), F))
    tetaF = np.dot(np.dot(np.dot(tetaF1, FT), Winv), Y)
print("МНК оцінка:", tetaF, end="\n\n")

# шукаємо тау0, тау
tau0 = np.dot(T, teta)
tau = np.dot(T, tetaF)

# обчислюємо матрицю V
FTF = np.linalg.pinv(np.dot(FT, F))
V = np.dot(np.dot(T, FTF), T.T)
Vinv = np.linalg.inv(V)

# обчислимо залишкову суму R0
YFteta = Y - np.dot(F, tetaF)
R0 =  np.dot(YFteta.T, YFteta) 

# обчислюємо статистику розподілену за законом Фішера,
# використовуючи МНК оцінки
taudiff = tau - tau0
taudiffV = np.dot(np.dot(taudiff.T, Vinv), taudiff)
taudiffVk = np.divide(taudiffV, k)
RNr = np.divide(R0, (N - r))
FisherA = np.divide(taudiffVk, RNr)
print("Значення статистик розподілених за законом Фішера")
print("a).Використовуючи МНК оцінку:\n", FisherA)

# обчислюємо статистику розподілену за законом Фішера,
# використовуючи залишкову суму квадратів
def func(teta):
    fr = 0
    for i in range(n):
        fr += (Y[i] - np.dot(F[i], teta))**2
    return fr
linear_constraint = scipy.optimize.LinearConstraint(T, tau0, tau0)
R1fr = scipy.optimize.minimize(func, teta, method = 'trust-constr',
                               constraints=linear_constraint)
R1 = R1fr.fun

if R0 <= R1:
    Rdiff = R1 - R0
    Rdiffk = np.divide(Rdiff, k)
    FisherB = np.divide(Rdiffk, RNr)
    print("б).Використовуючи залишкову суму квадратів:\n", FisherB, end="\n\n")
    
    # знаходимо критичні значення
    alpha1 = 0.05
    alpha2 = 0.01
    
    cv1 = scipy.stats.f.ppf(1 - alpha1, k, n-r)
    cv2 = scipy.stats.f.ppf(1 - alpha2, k, n-r) 
    
    print("Критичне значення при alpha = 0.05:", cv1)
    print("Критичне значення при alpha = 0.01:", cv2, end="\n\n")

# перевірка гіпотези
if FisherA <= abs(cv1):
    print("Гіпотеза прийнято")
else:
    print("Гіпотеза відхилено")
    
if FisherA <= abs(cv2):
    print("Гіпотеза прийнято")
else:
    print("Гіпотеза відхилено")
    
if FisherB <= abs(cv1):
    print("Гіпотеза прийнято")
else:
    print("Гіпотеза відхилено")
    
if FisherB <= abs(cv2):
    print("Гіпотеза прийнято")
else:
    print("Гіпотеза відхилено")
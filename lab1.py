import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x)

def g(x):
    return np.sin(x) * x

def num_deriv(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

def df(x):
    return -np.sin(x)

def dg(x):
    return np.cos(x) * x + np.sin(x)

def right_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def left_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


x = 1
h = 0.1
a = 0
b = np.pi/2
n = 1000

grid = np.linspace(-5, 5, 100)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(grid, f(grid), label='f(x)')
plt.plot(grid, df(grid), label="f'(x)")
plt.legend()
plt.title("График функции cos(x)")

plt.subplot(1, 2, 2)
plt.plot(grid, g(grid), label='g(x)')
plt.plot(grid, dg(grid), label="g'(x)")
plt.legend()
plt.title("График функции sin(x) * x")

plt.show()

x0 = 1
df_num = (f(x0 + h) - f(x0 - h)) / (2 * h)
dg_num = (g(x0 + h) - g(x0 - h)) / (2 * h)


x_range = (0, 1)

x_values = []
numerical_derivatives = []
exact_derivatives = []
for x in [i*h for i in range(int((x_range[1]-x_range[0])/h)+1)]:
    x_values.append(x)
    numerical_derivatives.append((f(x+h)-f(x))/h)
    exact_derivatives.append(df(x))

k = len(x_values)
sum_of_squares = sum([(numerical_derivatives[i]-exact_derivatives[i])**2 for i in range(k)])
sigma = math.sqrt(1/k * sum_of_squares)

def rmsd(f, f_prime, x, h):
    num = num_deriv(f, x, h)
    true = f_prime(x)
    return np.sqrt(np.mean((num - true)**2))

def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * np.sum(f(x[:-1]))

def trapezoid_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (np.sum(f(x[:-1])) + np.sum(f(x[1:])))/2

def simpson_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h/3 * np.sum(f(x[0:-2:2]) + 4*f(x[1::2]) + f(x[2::2]))

f_int = np.sin(np.pi**2)
g_int = np.pi

f_rect = rectangle_rule(f, a, np.pi**2, n)
f_trap = trapezoid_rule(f, a, np.pi**2, n)
f_simp = simpson_rule(f, a, np.pi**2, n)

g_rect = rectangle_rule(g, 0, np.pi, n)
g_trap = trapezoid_rule(g, 0, np.pi, n)
g_simp = simpson_rule(g, 0, np.pi, n)

# Подсчет истинного значения производной cos(x) на промежутке [0; 2pi]
web = np.linspace(0, 2 * np.pi, 100)
y_true = df(web)

# Подсчет значения производной cos(x) на промежутке [0; 2pi] с шагом h
h_values = [0.1, 0.05, 0.025, 0.0125]
rmsd_val = []
for h in h_values:
    y_numerical = central_difference(f, web, h)
    rmsd1 = np.sqrt(np.mean((y_numerical - y_true)**2))
    rmsd_val.append(rmsd1)

plt.plot(h_values, rmsd_val, 'o-')
plt.xscale('log')
plt.xlabel('Шаг h')
plt.ylabel('Отклонение от аналитического решения')
plt.title('Зависимость отклонения от величины шага')
plt.show()
print("------------Дифференцирование-------------")
print("Центральная разностная производная f'(1) = ", central_difference(f,x,h))
print("Левая разностная производная f'(1) =", left_diff(f, x, h))
print("Правая разностная производная f'(1) =", right_diff(f, x, h), "\n")

print("Центральная разностная производная g'(1) = ", central_difference(g,x,h))
print("Левая разностная производная g'(1) =", left_diff(g, x, h))
print("Правая разностная производная g'(1) =", right_diff(g, x, h), "\n")

print("Производная f(x) при x=1:", df_num)
print("Производная g(x) при x=1:", dg_num, "\n")

print("СКО: ", sigma, "\n")

print("------------Интегрирование-------------")
print("Определенный интеграл f(x) на промежутке [0; pi^2]:", f_int)
print("Интеграл f(x) формулой прямоугольников:", f_rect)
print("Интеграл f(x) формулой трапеций:", f_trap)
print("Интеграл f(x) формулой Симпсона:", f_simp, "\n")
print("Определенный интеграл g(x) на промежутке [0; pi]:", g_int)
print("Интеграл g(x) формулой прямоугольников:", g_rect)
print("Интеграл g(x) формулой трапеций:", g_trap)
print("Интеграл g(x) формулой Симпсона:", g_simp, "\n")

print("Погрешность между истинным значанием интеграла и найденным значением:")
print("Погрешность интегрирования f(x) формулой прямоугольников:", f"{np.abs(f_rect - f_int):.15f}")
print("Погрешность интегрирования f(x) формулой трапеций:", f"{np.abs(f_trap - f_int):.15f}")
print("Погрешность интегрирования f(x) формулой Симпсона:", f"{np.abs(f_simp - f_int):.15f}", "\n")
print("Погрешность интегрирования g(x) формулой прямоугольников:", f"{np.abs(g_rect - g_int):.15f}")
print("Погрешность интегрирования g(x) формулой трапеций:", f"{np.abs(g_trap - g_int):.15f}")
print("Погрешность интегрирования g(x) формулой Симпсона:", f"{np.abs(g_simp - g_int):.15f}")
print("----------------------------------------")
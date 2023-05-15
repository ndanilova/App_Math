import numpy as np
import matplotlib.pyplot as plt
from lab2.lab2 import golden_section_search, fibonacci_search


def f3(x): return 2 * x[0] ** 2 + 3 * x[1] ** 2


def df3(x): return np.array([4 * x[0], 6 * x[1]])


def f(x): return x[0] ** 2 + x[1] ** 2


def df(x): return np.array([2 * x[0], 2 * x[1]])


def f2(x): return (x[0] + 3) ** 2 - 10


def df2(x): return np.array([2 * (x[0] + 3),
                             0])


# Реализация градиентного спуска с постоянным шагом:
def gradient_descent_constant_step(f, df, x0, learning_rate, max_iterations=1000, tol=1e-6):
    x = x0
    trajectory = [x]
    fx = f(x)
    for i in range(max_iterations):
        x_new = x - learning_rate * df(x)
        trajectory.append(x_new)
        fx_new = f(x_new)
        if abs(fx_new - fx) < tol:
            break
        x = x_new
        fx = fx_new
    return np.array(trajectory)


def constant_step_contour_and_trajectory(starting_point, learning_rate):
    trajectory = gradient_descent_constant_step(f, df, starting_point, learning_rate)

    print("Gradient descent with a constant step")
    print("Minimum value of the function:", f(trajectory[-1]))
    print("Point of minimum value:", trajectory[-1])

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.contour(X, Y, Z, np.logspace(0, 5, 35))
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', 5, 3, 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient descent with a constant step")
    plt.show()


def gradient_descent_armijo(f, df, x0, alpha_init=1, tol=1e-6, max_iter=1000, c=0.4):
    x = x0
    trajectory = [x]
    fx = f(x)

    for i in range(max_iter):
        p = -df(x)
        alpha = alpha_init
        fx_new = f(x + alpha * p)
        while fx_new > fx + c * alpha * np.dot(-p, p):
            alpha /= 2
            fx_new = f(x + alpha * p)
        x_new = x + alpha * p
        trajectory.append(x_new)
        if abs(fx_new - fx) < tol:
            break
            x = x_new
            fx = fx_new
    return np.array(trajectory)


def armijo_step_contour_and_trajectory(starting_point):
    trajectory = gradient_descent_armijo(f3, df3, starting_point)

    print("Gradient descent with step splitting with Armiho condition")
    print("Minimum value of the function:", f(trajectory[-1]))
    print("Point of minimum value:", trajectory[-1])

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.contour(X, Y, Z, np.logspace(0, 5, 35))
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', 5, 3, 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient descent with step splitting with Armiho condition")
    plt.show()


def golden_ratio_search(a, b, tol, func):
    golden_ratio = (1 + 5 ** 0.5) / 2
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio
    fc = func(c)
    fd = func(d)
    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            c = b - (b - a) / golden_ratio
            fd = fc
            fc = func(c)
        else:
            a = c
            c = d
            d = a + (b - a) / golden_ratio
            fc = fd
            fd = func(d)

    return (a + b) / 2


def gradient_descent_golden_search(f, df, x0, tol=1e-6, max_iter=1000):
    x = x0
    trajectory = [x]
    fx = f(x)
    for i in range(max_iter):
        grad = df(x)

        def phi(alpha):
            return f(x - alpha * grad)

        alpha_min = golden_ratio_search(0, 1, 1e-1, phi)
        x_new = x - alpha_min * grad
        fx_new = f(x_new)
        if abs(fx_new - fx) < tol:
            break
        x = x_new
        fx = fx_new
        trajectory.append(x)
    return np.array(trajectory)


def golden_ratio_contour_and_trajectory(f, df, starting_point):
    trajectory = gradient_descent_golden_search(f, df, starting_point)
    print("The fastest gradient descent")
    print("Minimum value of the function:", f(trajectory[-1]))
    print("Point of minimum value:", trajectory[-1])
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.contour(X, Y, Z, np.logspace(0, 5, 35))
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', 5, 3, 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("The fastest gradient descent")
    plt.show()


def conjugate_gradient(f, df, x0, tol=1e-6, max_iter=1000000):
    x = x0
    trajectory = [x]
    g = df(x)
    fx = f(x)
    d = -g
    for i in range(max_iter):
        alpha = fibonacci_search(f, x, d, fx, g)
        x_new = x + alpha * d
        fx_new = f(x_new)
        g_new = df(x_new)
        if np.dot(g, g) == 0:
            beta = 0
        else:
            beta = np.dot(g_new, g_new - g) / np.dot(g, g)
        d_new = -g_new + beta * d
        if abs(fx_new - fx) < tol:
            break
        x, g, d, fx = x_new, g_new, d_new, fx_new
        trajectory.append(x)
    return np.array(trajectory)


def line_search(f, x, d, f_x, df_x, alpha_init=1, c=0.4):
    alpha = alpha_init
    while f(x + alpha * d) > f_x + c * alpha * np.dot(df_x, d):
        alpha /= 2
    return alpha


def conjugate_contour_and_trajectory(f, df, starting_point):
    trajectory = conjugate_gradient(f, df, starting_point)
    print("Сonjugate gradient method")
    print("Minimum value of the function:", f(trajectory[-1]))
    print("Point of minimum value:", trajectory[-1])
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.contour(X, Y, Z, np.logspace(0, 5, 35))
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', 5, 3, 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Сonjugate gradient method")
    plt.show()


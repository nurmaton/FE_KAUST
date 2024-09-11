import numpy as np
import matplotlib.pyplot as plt

def FiniteDifferenceMethod(a, b, alpha, beta, f, N):
    # Step size based on N subdivisions
    h = (b - a) / N  # step size
    x = np.linspace(a, b, N + 1)  # grid points including boundaries

    # Setting up the right-hand side vector f at the interior points
    fh = np.array([f(xi) for xi in x[1:-1]])

    # Creating the coefficient matrix A
    mainDiagonal = (2 * alpha / h**2 + beta) * np.ones(N - 1)
    offDiagonal = (-alpha / h**2) * np.ones(N - 2)
    
    # Creating the tridiagonal matrix A
    A = np.diag(mainDiagonal) + np.diag(offDiagonal, -1) + np.diag(offDiagonal, 1)

    # Solving the system A uh = fh using NumPy
    uh = np.linalg.solve(A, fh)

    # Adding boundary conditions u(a) = u(b) = 0
    u = np.zeros(N + 1)
    u[1:-1] = uh

    return x, u


# Example 1
a = 0  # left boundary
b = 1  # right boundary
alpha = 1  # coefficient for second derivative
beta = -225  # coefficient for u(x)
N = 16  # number of subdivisions (intervals)

# Define the forcing function f(x)
def f(x):
    return -225*x**2 + 225*x - 2

# Exact solution
x_exact = np.linspace(a, b, 200)
u_exact = x_exact**2 - x_exact

# Solve the equation
x, u = FiniteDifferenceMethod(a, b, alpha, beta, f, N)

# Plot the solution
plt.plot(x, u, label='Numerical solution', marker = 'o', ms = 3, c = 'navy')
plt.plot(x_exact, u_exact, label='Exact solution', linewidth = 1, c = 'm')
text = r"$\alpha = $" + f"{alpha}," + r"$\quad \beta = $" + f"{beta}," + r"$\quad N = $" + f"{N}"   
plt.text(.3, -0.05, text)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the 1D Laplace equation')
plt.legend()
plt.show()
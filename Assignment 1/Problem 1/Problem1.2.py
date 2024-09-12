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


# Example 2
a = 0  # left boundary
b = np.pi  # right boundary
alpha = -np.pi**2  # coefficient for u''(x)
beta = 10000*(2+np.sqrt(2))  # coefficient for u(x)
N = 100  # number of subdivisions (intervals)

# Checking condition (1.4)
h = (b - a) / N 
if beta*(4*alpha + beta*h**2) < 0:
    phi = np.arctan(np.sqrt(-beta*(4*alpha + beta*h**2))*h/(2*alpha + beta*h**2))
    if np.isclose(np.sin(phi*N),0):
        print("Avoiding singluarity issue by adding 1 to N")
        N = N + 1 

# Defining function f(x)
def f(x):
    return (10000*(2+np.sqrt(2))-np.pi**2)*np.sin(x)

# Exact solution
x_exact = np.linspace(a, b, 200)
u_exact = np.sin(x_exact)

# Solving the equation
x, u = FiniteDifferenceMethod(a, b, alpha, beta, f, N)

# Plotting the solution
plt.plot(x, u, label='Numerical solution', marker = 'o', ms = 3, c = 'navy')
plt.plot(x_exact, u_exact, label='Exact solution', linewidth = 1, c = 'm')
text = r"$\alpha \approx $" + f"{round(alpha,2)}," + r"$\quad \beta \approx $" + f"{round(beta,2)}," + r"$\quad N = $" + f"{N}"   
plt.text(0.8, 0.15, text)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the 1D Laplace equation')
plt.legend()
plt.show()

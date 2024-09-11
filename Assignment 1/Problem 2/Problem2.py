import numpy as np
import matplotlib.pyplot as plt


def FiniteDifferenceMethodNonHomogeneous(a, b, alpha, beta, f, N, chi, eta):    
    # Step size based on N subdivisions
    h = (b - a) / N  # step size
    x = np.linspace(a, b, N + 1)  # grid points including boundaries

    # Setting up the right-hand side vector f at the interior points
    fh = np.array([f(xi) for xi in x[1:-1]])
    fh[0] = fh[0] + alpha*chi/h**2
    fh[-1] = fh[-1] + alpha*eta/h**2

    # Creating the coefficient matrix A
    mainDiagonal = (2 * alpha / h**2 + beta) * np.ones(N - 1)
    offDiagonal = (-alpha / h**2) * np.ones(N - 2)
    
    # Creating the tridiagonal matrix A
    A = np.diag(mainDiagonal) + np.diag(offDiagonal, -1) + np.diag(offDiagonal, 1)

    # Solving the system A uh = fh using NumPy
    uh = np.linalg.solve(A, fh)

    # Adding boundary conditions u(a) = chi and u(b) = eta
    u = np.zeros(N + 1)
    u[0] = chi
    u[-1] = eta
    u[1:-1] = uh

    return x, u


# Example
a = 1  # left boundary
b = 2  # right boundary
alpha = 1 # coefficient for second derivative
beta = 1  # coefficient for u(x)
N = 15  # number of subdivisions (intervals)
chi = np.e  # non-homogeneous boundary at x = a
eta = 2*np.e**2  # non-homogeneous boundary at x = b

# Checking condition (1.4)
h = (b - a) / N 
if beta*(4*alpha + beta*h**2) < 0:
    phi = np.arctan(np.sqrt(-beta*(4*alpha + beta*h**2))*h/(2*alpha + beta*h**2))
    if np.isclose(np.sin(phi*N),0):
        print("Avoiding singluarity issue by adding 1 to N")
        N = N + 1 

# Define the forcing function f(x)
def f(x):
    return -2*np.exp(x)

# Exact solution
x_exact = np.linspace(a, b, 200)
u_exact = x_exact*np.exp(x_exact)

# Solve the equation
x, u = FiniteDifferenceMethodNonHomogeneous(a, b, alpha, beta, f, N, chi, eta)

# Plot the solution
plt.plot(x, u, label='Numerical solution', marker = 'o', ms = 3, c = 'navy')
plt.plot(x_exact, u_exact, label='Exact solution', linewidth = 1, c = 'm')
text = r"$\alpha = $" + f"{round(alpha,2)}," + r"$\quad \beta = $" + f"{round(beta,2)}," + r"$\quad N = $" + f"{N},"   + r"$\quad \chi \approx $" + f"{round(chi,2)}," + r"$\quad \eta \approx $" + f"{round(eta,2)}"  
plt.text(1.0, 12.5, text)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the 1D non-homogeneous Laplace equation')
plt.legend()
plt.show()
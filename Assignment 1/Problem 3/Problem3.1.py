import numpy as np
import matplotlib.pyplot as plt


def FiniteDifferenceMethodNeumann(a, b, alpha, beta, f, N, Lambda, Mu):
    # Case when beta is NOT zero
    if beta == 0:
        print("Beta should NOT be zero")
        exit(1)
        
    N = N - 1
    global counter
    counter = -1
    while True:
        N = N + 1
        counter += 1
        
        # Step size based on N subdivisions
        h = (b - a) / N  # step size
        
        # Creating the coefficient matrix A
        mainDiagonal = (2 * alpha / h**2 + beta) * np.ones(N - 1)
        offDiagonal = (-alpha / h**2) * np.ones(N - 2)
        
        # Creating the tridiagonal matrix A
        A = np.diag(mainDiagonal) + np.diag(offDiagonal, -1) + np.diag(offDiagonal, 1)
        A[0, 0] = alpha / h**2 + beta
        A[-1, -1] = alpha / h**2 + beta
        
        # Checking if matrix A is not singular
        if np.linalg.det(A) != 0:
            break

    x = np.linspace(a, b, N + 1)  # grid points including boundaries
    
    # Setting up the right-hand side vector f at the interior points
    fh = np.array([f(xi) for xi in x[1:-1]])
    fh[0] = fh[0] - alpha*Lambda/h
    fh[-1] = fh[-1] + alpha*Mu/h

    # Solving the system A uh = fh using NumPy
    uh = np.linalg.solve(A, fh)

    # Adding boundary conditions u(a) = chi and u(b) = eta
    u = np.zeros(N + 1)
    u[1:-1] = uh 
    u[0] = u[1] - Lambda*h  # from Neumann BC at x = a
    u[-1] = u[-2] + Mu*h  # from Neumann BC at x = b

    return x, u


# Example 1
a = 1  # left boundary
b = 2  # right boundary
alpha = 1 # coefficient for u''(x)
beta = 1 # coefficient for u(x)
N = 17  # number of subdivisions (intervals)
Lambda = 1  # Neumann BC at x = a (u'(a) = Lambda)
Mu = 4*np.log(2) + 2 # Neumann BC at x = b (u'(b) = Mu)


# Defining function f(x)
def f(x):
    return -2*np.log(x) - 3 + x**2*np.log(x)

# Exact solution
x_exact = np.linspace(a, b, 200)
u_exact = x_exact**2*np.log(x_exact)

# Solving the equation
x, u = FiniteDifferenceMethodNeumann(a, b, alpha, beta, f, N, Lambda, Mu)

# Plotting the solution
plt.plot(x, u, label='Numerical solution', marker = 'o', ms = 3, c = 'navy')
plt.plot(x_exact, u_exact, label='Exact solution', linewidth = 1, c = 'm')
text = r"$\alpha = $" + f"{round(alpha,2)}," + r"$\quad \beta = $" + f"{round(beta,2)}," + r"$\quad N = $" + f"{N + counter},"   + r"$\quad \lambda = $" + f"{round(Lambda,2)}," + r"$\quad \mu \approx $" + f"{round(Mu,2)}"  
plt.text(1.0, 2.5, text)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the 1D Laplace equation witn Neumann BC\nForward and Backward Differences')
plt.legend()
plt.show()

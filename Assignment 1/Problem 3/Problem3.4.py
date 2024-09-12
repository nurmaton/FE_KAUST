import numpy as np
import matplotlib.pyplot as plt
import warnings

def FiniteDifferenceMethodNeumann(a, b, alpha, beta, f, N, Lambda, Mu): 
    if beta == 0:
        # Setting perturbation (epsilon) that will be added to the diagonal elements of matrix A
        epsilon = 0.14
    else:
        epsilon = 0.0    
    
    N = N - 1
    global counter
    counter = -1
    while True:
        N = N + 1
        counter += 1
        
        # Step size based on N subdivisions
        h = (b - a) / N  # step size
        
        # Creating the coefficient matrix A
        mainDiagonal = (2 * alpha / h**2 + beta + epsilon) * np.ones(N + 1)
        offDiagonal = (-alpha / h**2) * np.ones(N)
        
        # Creating the tridiagonal matrix A
        A = np.diag(mainDiagonal) + np.diag(offDiagonal, -1) + np.diag(offDiagonal, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        
        print(A)
        
        # Checking if matrix A is not singular
        if np.linalg.det(A) != 0:
            if counter != 0:
                print(f"Avoiding singluarity issue by adding {counter} to N")
            break
    
    x = np.linspace(a, b, N + 1)  # grid points including boundaries
    
    # Setting up the right-hand side vector f at the interior points
    fh = np.array([f(xi) for xi in x])
    fh[0] = fh[0] - 2*alpha*Lambda/h
    fh[-1] = fh[-1] + 2*alpha*Mu/h

    # Solving the system A uh = fh using NumPy
    uh = np.linalg.solve(A, fh)

    return x, uh


# Example 4
a = -1  # left boundary
b = 1  # right boundary
alpha = -1 # coefficient for u''(x)
beta = 0 # coefficient for u(x)
N = 17 # number of subdivisions (intervals)
Lambda = -4  # Neumann BC at x = a (u'(a) = Lambda)
Mu = 4 # Neumann BC at x = b (u'(b) = Mu)


# Defining function f(x)
def f(x):
    return 12*x**2

# Modifying the boundary in cases where |f(x)| tends to infinity at a boundary point
eps = 0.107*(b-a)/N
with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        res = f(a)
    except Warning:
        a = a + eps
    except ZeroDivisionError:
        a = a + eps
        
with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        res = f(b)
    except Warning:
        b = b - eps
    except ZeroDivisionError:
        b = b - eps

# Exact solution
x_exact = np.linspace(a, b, 200)
u_exact = x_exact**4 

# Solving the equation
x, u = FiniteDifferenceMethodNeumann(a, b, alpha, beta, f, N, Lambda, Mu)

# Plotting the solution
plt.plot(x, u, label='Numerical solution', marker = 'o', ms = 3, c = 'navy')
plt.plot(x_exact, u_exact, label='Exact solution', linewidth = 1, c = 'm')
text = r"$\alpha = $" + f"{round(alpha,2)}," + r"$\quad \beta = $" + f"{round(beta,2)}," + r"$\quad N = $" + f"{N + counter},"   + r"$\quad \lambda = $" + f"{round(Lambda,2)}," + r"$\quad \mu = $" + f"{round(Mu,2)}"  
plt.text(-0.5, 0.8, text)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the 1D Laplace equation witn Neumann BC\nCentral Difference')
plt.legend()
plt.show()

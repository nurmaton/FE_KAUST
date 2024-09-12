import numpy as np
import matplotlib.pyplot as plt
from random import seed, random

def FiniteDifferenceMethodNonUniform(alpha, beta, f, subdivision, chi, eta):
    N = len(subdivision) - 1  # number of subdivisions (intervals)
    h = np.diff(subdivision)  # step sizes (hi)

    # Setting up the right-hand side vector f at the interior points
    fh = np.array([f(xi) for xi in subdivision[1:-1]])
    fh[0] = fh[0] + 2*alpha*chi/(h[0]*(h[0] + h[1]))
    fh[-1] = fh[-1] + 2*alpha*eta/(h[-1]*(h[-2] + h[-1]))

    # Creating the coefficient matrix A for non-uniform grid
    A = np.zeros((N-1, N-1))
    for i in range(N-1):
        if i > 0:
            A[i, i-1] = -2*alpha / (h[i] * (h[i] + h[i+1]))
        A[i, i] = 2 * alpha / (h[i] * h[i+1]) + beta
        if i < N-2:
            A[i, i+1] = -2*alpha / (h[i+1] * (h[i] + h[i+1]))

    # Solving the system A uh = fh using NumPy
    uh = np.linalg.solve(A, fh)

    # Adding boundary conditions u(a) = chi and u(b) = eta
    u = np.zeros(N + 1)
    u[0] = chi
    u[-1] = eta
    u[1:-1] = uh

    return subdivision, u

def generateRandomSubdivision(a, b, N):
    # seed random number generator
    seed(2)
    # generating random numbers between [a,b]
    randomPoints = []
    subdivision = np.zeros(N+1)
    subdivision[0] = a
    subdivision[-1] = b
    for _ in range(N-1):
        while True:
            randomValue = a + (b-a)*random()
            if randomValue != a and randomValue not in randomPoints:
                break 
        randomPoints.append(randomValue)
    subdivision[1:-1] = np.sort(randomPoints) 
    
    return subdivision


# Example 3
a = 1  # left boundary
b = 2  # right boundary
alpha = 1 # coefficient for u''(x)
beta = 1  # coefficient for u(x)
N = 15  # number of subdivisions (intervals)
chi = np.e  # non-homogeneous boundary at x = a
eta = 2*np.e**2  # non-homogeneous boundary at x = b

# Creating subdivision
subdivision = generateRandomSubdivision(a, b, N)
print(subdivision)

# Defining function f(x)
def f(x):
    return -2*np.exp(x)

# Exact solution
x_exact = np.linspace(a, b, 200)
u_exact = x_exact*np.exp(x_exact)

# Solving the equation
x, u = FiniteDifferenceMethodNonUniform(alpha, beta, f, subdivision, chi, eta)

# Plotting the solution
plt.plot(x, u, label='Numerical solution', marker = 'o', ms = 3, c = 'navy')
plt.plot(x_exact, u_exact, label='Exact solution', linewidth = 1, c = 'm')
text = r"$\alpha = $" + f"{round(alpha,2)}," + r"$\quad \beta = $" + f"{round(beta,2)}," + r"$\quad N = $" + f"{N},"   + r"$\quad \chi \approx $" + f"{round(chi,2)}," + r"$\quad \eta \approx $" + f"{round(eta,2)}"  
plt.text(1.0, 12.5, text)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the 1D non-homogeneous Laplace equation\nWith Non-Uniform Subdivision')
plt.legend()
plt.show()
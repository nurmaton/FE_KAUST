import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def FiniteDifferenceMethod2D(a, b, alpha, beta, f, N):
    # Defining the grid
    h = (b - a) / N  # Grid spacing
    
    # Grid points including boundaries
    x = np.linspace(a, b, N + 1)  
    y = np.linspace(a, b, N + 1)
    
    # Total number of interior points (excluding boundary points)
    numOfInteriorPoints = (N - 1) ** 2

    # Initializing the matrix A and vector fh
    A = np.zeros((numOfInteriorPoints, numOfInteriorPoints))
    fh = np.zeros(numOfInteriorPoints)  # Right-hand side vector

    # Filling the right-hand side vector with the forcing function values
    indexMap = {}  # To map the 2D index (i, j) to the 1D index in A and fh
    count = 0
    for i in range(1, N):
        for j in range(1, N):
            indexMap[(i, j)] = count
            fh[count] = f(x[i], y[j])
            count += 1

    # Filling the matrix A using finite difference coefficients
    for i in range(1, N):
        for j in range(1, N):
            index = indexMap[(i, j)]
            A[index, index] = 4 * alpha / h**2 + beta  # Diagonal element
            if (i-1, j) in indexMap:  # Left neighbor
                A[index, indexMap[(i-1, j)]] = -alpha / h**2
            if (i+1, j) in indexMap:  # Right neighbor
                A[index, indexMap[(i+1, j)]] = -alpha / h**2
            if (i, j-1) in indexMap:  # Bottom neighbor
                A[index, indexMap[(i, j-1)]] = -alpha / h**2
            if (i, j+1) in indexMap:  # Top neighbor
                A[index, indexMap[(i, j+1)]] = -alpha / h**2

    # Solving the system A uh = fh using NumPy
    uh = np.linalg.solve(A, fh)

    # Reshaping the solution to a 2D array and include boundary values
    u = np.zeros((N + 1, N + 1))
    count = 0
    for i in range(1, N):
        for j in range(1, N):
            u[i, j] = uh[count]
            count += 1

    return x, y, u

# Example
a, b = 0, 1  # Domain boundaries
N = 50  # Number of subdivisions in each direction
alpha = 1  # coefficient for Delta u
beta = 1  # coefficient for u

# Defining function f(x, y)
def f(x, y):
    return (x**2 - x) * (y**2 - y)- 2 * (x**2 + y**2 - x - y)

# Exact solution
x_exact = np.linspace(a, b, 200)
y_exact = np.linspace(a, b, 200)
X_exact, Y_exact = np.meshgrid(x_exact, y_exact)
u_exact = X_exact*Y_exact*(X_exact-1)*(Y_exact-1)

# Solving the equation
x, y, u = FiniteDifferenceMethod2D(a, b, alpha, beta, f, N)

# Plotting the solution in 3D
X, Y = np.meshgrid(x, y)
X_exact, Y_exact = np.meshgrid(x_exact, y_exact)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, u, cmap="tab20b_r")
ax.plot_surface(X_exact, Y_exact, u_exact, cmap="RdGy")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y)")
# ax.set_title("Solution of the 2D Laplace equation witn Dirichlet BC\n Numerical Solution")
ax.set_title("Solution of the 2D Laplace equation witn Dirichlet BC\n Exact Solution")
plt.show()
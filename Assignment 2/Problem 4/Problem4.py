import numpy as np
import matplotlib.pyplot as plt

def finite_element_method(N):
    """
    Solves the boundary value problem -u''(x) = 1 with boundary conditions u(0) = 0 and u'(1) = 0
    using the Finite Element Method.
    
    Parameters:
    N (int): Number of elements (subintervals in the mesh)

    Returns:
    x (ndarray): Mesh points (including boundaries)
    U (ndarray): Approximated solution at mesh points
    """
    # Element size (step length)
    h = 1.0 / N
    
    # Generating mesh points from 0 to 1 (inclusive)
    x = np.linspace(0, 1, N + 1)  # N+1 points including boundaries
    
    # Initializing stiffness matrix A and load vector b
    A = np.zeros((N, N))  # Stiffness matrix (tridiagonal)
    b = np.zeros(N)       # Load vector
    
    # Creating diagonal and off-diagonal entries for matrix A
    main_diagonal = (2 / h) * np.ones(N)
    off_diagonal = (-1 / h) * np.ones(N - 1)
    
    # Building the tridiagonal matrix A
    A = np.diag(main_diagonal) + np.diag(off_diagonal, -1) + np.diag(off_diagonal, 1)
    
    # Filling load vector b with the right-hand side values
    b = h * np.ones(N)
    
    # Adjusting for Neumann boundary condition at the last node
    A[-1, -1] = 1 / h  # Modifying last diagonal element of A for Neumann boundary
    b[-1] = h / 2      # Modifying last element of b for the boundary condition
    
    # Solving the system of linear equations A * U_1_N = b
    U_1_N = np.linalg.solve(A, b)
    
    # Completing solution: Include Dirichlet boundary condition u(0) = 0
    U = np.zeros(N + 1)  # Initializing solution vector, with u(0) already 0
    U[1:] = U_1_N        # Filling in the solution starting from u(1) onwards
    
    return x, U

def exact_solution():
    """
    Returns the exact solution for comparison: u(x) = x - 0.5 * x^2.

    Returns:
    x (ndarray): Dense mesh points for the exact solution
    u (ndarray): Exact solution values at the mesh points
    """
    x = np.linspace(0, 1, 200)  # Generating a denser set of points for the exact solution
    u = x - 0.5 * x**2          # Exact solution to the boundary value problem
    return x, u

def main():
    """
    Main function to solve the BVP using FEM and plot the FEM and exact solutions.
    """
    N = 11  # Number of elements (subintervals in the mesh)
    
    # Solving using FEM
    x, U = finite_element_method(N)
    
    # Getting the exact solution for comparison
    x_exact, u_exact = exact_solution()
    
    # Visualization of the FEM and exact solutions
    plt.plot(x, U, label='FEM Solution', marker='o', ms=3, c='navy')   # FEM solution (with markers)
    plt.plot(x_exact, u_exact, label='Exact Solution', linewidth=1, c='m')  # Exact solution
    
    # Adding label for the number of elements
    text = r"$N = $" + f"{N}"
    plt.text(0.4, 0.5, text)
    
    # Setting plot labels and title
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution of -u\'\'(x) = 1 with u(0) = 0, u\'(1) = 0')
    
    # Adding legend
    plt.legend()
    
    # Displaying the plot
    plt.show()

if __name__ == '__main__':
    main()

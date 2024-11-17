import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Define the function f(x) to be projected onto the finite element space.
    
    Args:
        x (float): The point at which to evaluate the function.
    
    Returns:
        float: The value of f(x) at the point x.
    """
    return np.sin(2 * np.pi * x)  # Example function

def phi_i(x, i, x_vals):
    """Evaluate the piecewise linear basis function phi_i at point x.
    
    Args:
        x (float): The point at which to evaluate the basis function.
        i (int): Index of the basis function.
        x_vals (numpy.ndarray): Array of mesh points in the interval [0, 1].
    
    Returns:
        float: The value of the basis function phi_i at point x.
    """
    h = x_vals[1] - x_vals[0]  # Mesh size (assuming uniform mesh)
    if i == 0:  # Left boundary basis function phi_0
        if x_vals[0] <= x <= x_vals[1]:
            return (x_vals[1] - x) / h
    elif i == len(x_vals) - 1:  # Right boundary basis function phi_N
        if x_vals[-2] <= x <= x_vals[-1]:
            return (x - x_vals[-2]) / h
    else:  # Interior basis functions
        if x_vals[i - 1] <= x <= x_vals[i]:
            return (x - x_vals[i - 1]) / h
        elif x_vals[i] <= x <= x_vals[i + 1]:
            return (x_vals[i + 1] - x) / h
    return 0.0

def gauss_legendre_2_point(f, phi_i, a, b, i, x_vals):
    """Apply 2-point Gauss-Legendre quadrature on the interval [a, b] for the integrand f(x) * phi_i(x).
    
    Args:
        f (function): Function to be integrated.
        phi_i (function): Basis function to multiply with f(x).
        a (float): Start of the interval.
        b (float): End of the interval.
        i (int): Index of the basis function phi_i.
        x_vals (numpy.ndarray): Array of mesh points in the interval [0, 1].
    
    Returns:
        float: Approximation of the integral of f(x) * phi_i(x) over [a, b].
    """
    nodes = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    weights = [1, 1]
    transformed_nodes = [(b - a) / 2 * node + (a + b) / 2 for node in nodes]
    integral = (b - a) / 2 * sum(weights[j] * f(transformed_nodes[j]) * phi_i(transformed_nodes[j], i, x_vals) for j in range(2))
    return integral

def assemble_load_vector(N):
    """Assemble the load vector b for the L2 projection problem.
    
    Args:
        N (int): Number of intervals in the mesh.
    
    Returns:
        numpy.ndarray: The load vector b, where each entry b[i] approximates the integral of f(x) * phi_i(x).
    """
    x_vals = np.linspace(0, 1, N + 1)  # Mesh points
    h = x_vals[1] - x_vals[0]
    b = np.zeros(N + 1)  # Initialize the load vector

    for i in range(N + 1):
        if i == 0:  # Left boundary node
            b[i] = gauss_legendre_2_point(f, phi_i, x_vals[0], x_vals[1], i, x_vals)
        elif i == N:  # Right boundary node
            b[i] = gauss_legendre_2_point(f, phi_i, x_vals[N - 1], x_vals[N], i, x_vals)
        else:  # Interior nodes
            b[i] = (gauss_legendre_2_point(f, phi_i, x_vals[i - 1], x_vals[i], i, x_vals) +
                    gauss_legendre_2_point(f, phi_i, x_vals[i], x_vals[i + 1], i, x_vals))
    return b

def assemble_mass_matrix(N):
    """Assemble the mass matrix M for the finite element space.
    
    Args:
        N (int): Number of intervals in the mesh.
    
    Returns:
        numpy.ndarray: The mass matrix M.
    """
    h = 1.0 / N  # Element size (step length)
    
    # Create diagonal and off-diagonal entries for the mass matrix M
    main_diagonal = (2 * h / 3) * np.ones(N + 1)  # Interior diagonal values set to 2h/3
    off_diagonal = (h / 6) * np.ones(N)  # Off-diagonal values set to h/6
    
    # Construct the matrix with main and off-diagonals
    M = np.diag(main_diagonal) + np.diag(off_diagonal, -1) + np.diag(off_diagonal, 1)
    
    # Adjust the boundary diagonal entries
    M[0, 0] = h / 3
    M[-1, -1] = h / 3
    return M

def l2_projection_finite_element_method(N):
    """Compute the L2 projection of f onto the finite element space V_h.
    
    Args:
        N (int): Number of intervals in the mesh.
    
    Returns:
        numpy.ndarray: Projection coefficients for each basis function in V_h.
    """
    M = assemble_mass_matrix(N)  # Assemble mass matrix
    b = assemble_load_vector(N)  # Assemble load vector
    alpha = np.linalg.solve(M, b)  # Solve for projection coefficients
    return alpha

def main():
    """Main function to plot the original function f(x) and its L2 projection onto V_h."""
    N = 10  # Number of intervals
    x_vals = np.linspace(0, 1, N + 1)  # Mesh points
    alpha = l2_projection_finite_element_method(N)  # Compute projection coefficients
    
    # Fine grid for plotting
    x_fine = np.linspace(0, 1, 100)
    y_true = f(x_fine)
    
    # Compute the projection on the fine grid using alpha
    y_proj = np.zeros_like(x_fine)
    for i in range(N + 1):
        y_proj += alpha[i] * np.array([phi_i(x, i, x_vals) for x in x_fine])
    
    # Plot the original function, projection, and projection values at the nodes
    plt.plot(x_fine, y_true, label='Original function f(x)', c='navy')
    plt.plot(x_fine, y_proj, label=f'$L^2$ projection $\Pi$f(x)', linestyle='--', c='m')
    plt.scatter(x_vals, alpha, color='m', label='Projection at nodes', s=15)
    
    # Adding label for the number of elements
    text = r"$N = $" + f"{N}"
    plt.text(0.8, 0.5, text)
    
    # Setting plot labels, title, and grid
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'$L^2$ Projection of f onto $V_h$')
    plt.legend()
    
    # Display the plot
    plt.show()

# Run the main function if the script is executed
if __name__ == '__main__':
    main()

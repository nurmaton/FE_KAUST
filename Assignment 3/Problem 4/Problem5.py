import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

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

def compute_projection(x, alpha, x_vals):
    """Compute the projected function Pf at any point x using coefficients alpha.
    
    This function evaluates the projected function Pf(x) at a specific point x, 
    based on the piecewise linear basis functions and the projection coefficients alpha.
    
    Args:
        x (float): The point at which to evaluate the projection Pf.
        alpha (numpy.ndarray): Array of projection coefficients for each basis function.
        x_vals (numpy.ndarray): Array of mesh points in the interval [0, 1].

    Returns:
        float: The value of the projected function Pf at the point x.
    """
    N = len(x_vals) - 1  # Number of intervals
    # Compute the projection as the sum of alpha[i] * phi_i(x) over all basis functions phi_i
    projection = sum(alpha[i] * phi_i(x, i, x_vals) for i in range(N + 1))
    return projection

def compute_l2_error(N):
    """Compute the L2 error norm between the original function f and its projection Pf.
    
    This function calculates the L2 norm of the error e(x) = f(x) - Pf(x) over the interval [0, 1].
    It first computes the projection coefficients, then defines the error function and 
    integrates its square over [0, 1] using `quad`. The square root of this integral gives the L2 error norm.
    
    Args:
        N (int): Number of intervals in the mesh.

    Returns:
        float: The L2 error norm between f and its projection Pf.
    """
    x_vals = np.linspace(0, 1, N + 1)  # Generate mesh points in [0, 1]
    alpha = l2_projection_finite_element_method(N)  # Calculate projection coefficients

    # Define the error function e(x) = f(x) - Pf(x)
    error_func = lambda x: f(x) - compute_projection(x, alpha, x_vals)
    
    # Integrate the squared error over [0, 1] using `quad` to approximate the L2 error
    l2_error_squared, _ = quad(lambda x: error_func(x)**2, 0, 1)
    
    # Take the square root to obtain the L2 norm
    l2_error = np.sqrt(l2_error_squared)
    return l2_error

def main():
    """Main function to estimate the L2 error and plot convergence."""
    # Define different values of N (number of intervals) to test convergence with decreasing h
    N_values = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]  # List of intervals for different mesh sizes
    h_values = [1.0 / N for N in N_values]  # Calculate mesh size h for each N
    
    # Calculate the L2 error for each N, storing results in the errors list
    errors = [compute_l2_error(N) for N in N_values]  # L2 errors for each mesh size
    
    # Plot the errors on a log-log scale to observe convergence rate
    plt.loglog(h_values, errors, '-o', label=r'$L^2$ error $\|f - \Pi f\|_{L^2}$', c='m', ms=5)
    
    # Plot the theoretical convergence rate (O(h^2)) as a reference line
    plt.loglog(h_values, [h**2 for h in h_values], '--', label=r'$\mathcal{O}(h^2)$', c='navy')
    
    # Calculate the average slope using consecutive points on the log-log scale
    slopes = [
        (np.log(errors[i+1]) - np.log(errors[i])) / (np.log(h_values[i+1]) - np.log(h_values[i]))
        for i in range(len(h_values) - 1)
    ]
    average_slope = np.mean(slopes)  # Average of all slopes
    
    # Display the average slope on the plot
    plt.text(h_values[2], errors[0], f"Avg Slope $\\approx$ {average_slope:.2f}", fontsize=10, color="red")
    
    # Label the axes
    plt.xlabel('Mesh size h (log scale)')
    plt.ylabel(f'$L^2$ Error (log scale)')
    
    # Add legend and title
    plt.legend()
    plt.title(f'$L^2$ Error Convergence for $L^2$ Projection (log-log scale)')
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
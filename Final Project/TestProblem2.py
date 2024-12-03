import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


class Mesh:
    """
    A class to generate a 2D mesh for the domain using Delaunay triangulation.

    Attributes:
        triangulation (Delaunay): Delaunay triangulation of the mesh points.
        boundary_points (ndarray): Indices of the boundary nodes.
        bc_points (dict): Dictionary for boundary conditions.
    """

    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y):
        """
        Initialize the Mesh object.

        Args:
            x_min (float): Minimum x-coordinate.
            x_max (float): Maximum x-coordinate.
            n_x (int): Number of points in x-direction.
            y_min (float): Minimum y-coordinate.
            y_max (float): Maximum y-coordinate.
            n_y (int): Number of points in y-direction.
        """
        # Create a list with points coordinates (x, y)
        points = []
        nodes_x = np.linspace(x_min, x_max, n_x)
        nodes_y = np.linspace(y_min, y_max, n_y)
        for nx in nodes_x:
            for ny in nodes_y:
                points.append([nx, ny])
        points = np.array(points)

        # Create Delaunay triangulation object
        self.triangulation = Delaunay(points)

        # Identify the boundary points (nodes on the convex hull)
        self.boundary_points = np.unique(self.triangulation.convex_hull.flatten())

        # Initialize the boundary conditions dictionary
        self.bc_points = {
            "dirichlet": dict()
        }


class ElementTriangle:
    """
    A class representing a standard triangle element with linear basis functions.
    """

    def __init__(self):
        """
        Initialize the ElementTriangle object.
        """
        pass  # No initialization needed for this class

    @staticmethod
    def twice_Ae(p1, p2, p3):
        """
        Calculate twice the area of the triangle element.

        Args:
            p1, p2, p3 (array_like): Coordinates of the triangle vertices.

        Returns:
            float: Twice the area of the triangle.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    def grad_xy_psi1(self, p1, p2, p3):
        """
        Compute the gradient of basis function psi1 in global coordinates.

        Args:
            p1, p2, p3 (array_like): Coordinates of the triangle vertices.

        Returns:
            ndarray: Gradient vector of psi1.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        return np.array([y2 - y3, x3 - x2]) / self.twice_Ae(p1, p2, p3)

    def grad_xy_psi2(self, p1, p2, p3):
        """
        Compute the gradient of basis function psi2 in global coordinates.

        Args:
            p1, p2, p3 (array_like): Coordinates of the triangle vertices.

        Returns:
            ndarray: Gradient vector of psi2.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        return np.array([y3 - y1, x1 - x3]) / self.twice_Ae(p1, p2, p3)

    def grad_xy_psi3(self, p1, p2, p3):
        """
        Compute the gradient of basis function psi3 in global coordinates.

        Args:
            p1, p2, p3 (array_like): Coordinates of the triangle vertices.

        Returns:
            ndarray: Gradient vector of psi3.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        return np.array([y1 - y2, x2 - x1]) / self.twice_Ae(p1, p2, p3)

    @staticmethod
    def psi1(xi, eta):
        """
        Basis function psi1 in reference coordinates.

        Args:
            xi (float): Local coordinate xi.
            eta (float): Local coordinate eta.

        Returns:
            float: Value of psi1 at (xi, eta).
        """
        return 1 - xi - eta

    @staticmethod
    def psi2(xi, eta):
        """
        Basis function psi2 in reference coordinates.

        Args:
            xi (float): Local coordinate xi.
            eta (float): Local coordinate eta.

        Returns:
            float: Value of psi2 at (xi, eta).
        """
        return xi

    @staticmethod
    def psi3(xi, eta):
        """
        Basis function psi3 in reference coordinates.

        Args:
            xi (float): Local coordinate xi.
            eta (float): Local coordinate eta.

        Returns:
            float: Value of psi3 at (xi, eta).
        """
        return eta

    def get_xy(self, xi, eta, p1, p2, p3):
        """
        Coordinate transformation from local (xi, eta) to global (x, y).

        Args:
            xi (float): Local coordinate xi.
            eta (float): Local coordinate eta.
            p1, p2, p3 (array_like): Global coordinates of the triangle vertices.

        Returns:
            tuple: Global coordinates (x, y).
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        x = (x1 * self.psi1(xi, eta) +
             x2 * self.psi2(xi, eta) +
             x3 * self.psi3(xi, eta))

        y = (y1 * self.psi1(xi, eta) +
             y2 * self.psi2(xi, eta) +
             y3 * self.psi3(xi, eta))
        
        return x, y


class QuadratureFormula:
    """
    A class for numerical integration over a triangle using quadrature.
    """

    def __init__(self):
        """
        Initialize the QuadratureFormula object.
        """
        # Quadrature points and weights
        self.quadrature_points = [
            (1/3, 1/3),
            (2/15, 2/15),
            (11/15, 2/15),
            (2/15, 11/15)
        ]
        self.weights = [-27/96, 25/96, 25/96, 25/96]
        self.element_triangle = ElementTriangle()

    def calculate_local_load_vector(self, _function_f, p1, p2, p3):
        """
        Calculate the local load vector for an element.

        Args:
            _function_f (function): Right-hand side function f(x, y).
            p1, p2, p3 (array_like): Global coordinates of the triangle vertices.

        Returns:
            ndarray: Local load vector (size 3).
        """
        # Get the global (x, y) coordinates at the quadrature points
        xy_coordinates = [self.element_triangle.get_xy(
            quadrature_point[0], quadrature_point[1], p1, p2, p3)
            for quadrature_point in self.quadrature_points]

        twiceArea = abs(self.element_triangle.twice_Ae(p1, p2, p3))
        
        # Initialize local load vector
        F_local = np.zeros(3)

        # Compute entries of the local load vector using quadrature
        for i in range(3):
            F_local[i] = sum([
                weight * _function_f(xy[0], xy[1]) *
                getattr(self.element_triangle, f'psi{i+1}')(
                    quadrature_point[0], quadrature_point[1])
                for weight, quadrature_point, xy in zip(self.weights, self.quadrature_points, xy_coordinates)
            ]) * twiceArea

        return F_local

    def calculate_local_mass_matrix(self, _function_q, p1, p2, p3):
        """
        Calculate the local mass matrix for an element.

        Args:
            _function_q (function): Coefficient function q(x, y).
            p1, p2, p3 (array_like): Global coordinates of the triangle vertices.

        Returns:
            ndarray: Local mass matrix (size 3x3).
        """
        # Get the global (x, y) coordinates at the quadrature points
        xy_coordinates = [self.element_triangle.get_xy(
            quadrature_point[0], quadrature_point[1], p1, p2, p3)
            for quadrature_point in self.quadrature_points]

        # Initialize local mass matrix
        M_local = np.zeros((3, 3))
        twiceArea = abs(self.element_triangle.twice_Ae(p1, p2, p3))

        # Compute entries of the local mass matrix using quadrature
        for i in range(3):
            for j in range(i, 3):
                M_local[i, j] = sum([
                    weight * _function_q(xy[0], xy[1]) *
                    getattr(self.element_triangle, f'psi{i+1}')(
                        quadrature_point[0], quadrature_point[1]) *
                    getattr(self.element_triangle, f'psi{j+1}')(
                        quadrature_point[0], quadrature_point[1])
                    for weight, quadrature_point, xy in zip(self.weights, self.quadrature_points, xy_coordinates)
                ]) * twiceArea
                M_local[j, i] = M_local[i, j]  # Symmetry

        return M_local

    def calculate_local_stiffness_matrix(self, _function_p, p1, p2, p3):
        """
        Calculate the local stiffness matrix for an element.

        Args:
            _function_p (function): Coefficient function p(x, y).
            p1, p2, p3 (array_like): Global coordinates of the triangle vertices.

        Returns:
            ndarray: Local stiffness matrix (size 3x3).
        """
        # Get the global (x, y) coordinates at the quadrature points
        xy_coordinates = [self.element_triangle.get_xy(
            quadrature_point[0], quadrature_point[1], p1, p2, p3)
            for quadrature_point in self.quadrature_points]

        # Initialize local stiffness matrix
        A_local = np.zeros((3, 3))
        twiceArea = abs(self.element_triangle.twice_Ae(p1, p2, p3))

        # Gradients of basis functions (constant over the element)
        grad_xy_psi = [
            self.element_triangle.grad_xy_psi1(p1, p2, p3),
            self.element_triangle.grad_xy_psi2(p1, p2, p3),
            self.element_triangle.grad_xy_psi3(p1, p2, p3)
        ]

        # Compute entries of the local stiffness matrix using quadrature
        for i in range(3):
            for j in range(i, 3):
                A_local[i, j] = sum([
                    weight * _function_p(xy[0], xy[1]) *
                    np.dot(grad_xy_psi[i], grad_xy_psi[j])
                    for weight, xy in zip(self.weights, xy_coordinates)
                ]) * twiceArea
                A_local[j, i] = A_local[i, j]  # Symmetry

        return A_local


class FESelfAdjointEllipticPDE2D:
    """
    A class to solve the 2D self-adjoint elliptic PDE using the finite element method.
    """

    def __init__(self, _mesh, _function_p, _function_q, _function_f):
        """
        Initialize the FESelfAdjointEllipticPDE2D solver.

        Args:
            _mesh (Mesh): Mesh object.
            _function_p (function): Coefficient function p(x, y).
            _function_q (function): Coefficient function q(x, y).
            _function_f (function): Right-hand side function f(x, y).
        """
        self.get_triangle_element = ElementTriangle()
        self.quadrature_formula = QuadratureFormula()

        self.mesh = _mesh
        self.n_elements = self.mesh.triangulation.nsimplex
        self.n_points = self.mesh.triangulation.npoints

        self.function_f = _function_f
        self.function_p = _function_p
        self.function_q = _function_q

        # Initialize global matrices and vectors
        self.A = np.zeros((self.n_points, self.n_points))
        self.M = np.zeros((self.n_points, self.n_points))
        self.F = np.zeros((self.n_points, 1))
        self.u = np.zeros_like(self.F)

    def calculate_local_A_M_b(self, p1, p2, p3):
        """
        Calculate local stiffness matrix, mass matrix, and load vector.

        Args:
            p1, p2, p3 (array_like): Global coordinates of the triangle vertices.

        Returns:
            tuple: Local stiffness matrix, mass matrix, and load vector.
        """
        A_local = self.quadrature_formula.calculate_local_stiffness_matrix(
            self.function_p, p1, p2, p3)
        M_local = self.quadrature_formula.calculate_local_mass_matrix(
            self.function_q, p1, p2, p3)
        F_local = self.quadrature_formula.calculate_local_load_vector(
            self.function_f, p1, p2, p3)

        return A_local, M_local, F_local

    def set_A_M_b(self):
        """
        Assemble the global stiffness matrix, mass matrix, and load vector.
        """
        for i, el_ps in enumerate(self.mesh.triangulation.simplices):
            # Extract element's nodes
            p1, p2, p3 = (self.mesh.triangulation.points[el_ps[0]],
                          self.mesh.triangulation.points[el_ps[1]],
                          self.mesh.triangulation.points[el_ps[2]])

            # Compute local matrices and vector
            A_local, M_local, F_local = self.calculate_local_A_M_b(
                p1, p2, p3)

            # Assemble into global matrices and vector
            for i_local, i_global in enumerate(el_ps):
                self.F[i_global, 0] += F_local[i_local]
                for j_local, j_global in enumerate(el_ps):
                    self.A[i_global, j_global] += A_local[i_local, j_local]
                    self.M[i_global, j_global] += M_local[i_local, j_local]

    def set_boundary_conditions(self):
        """
        Apply Dirichlet boundary conditions.
        """
        # Modify the right-hand side vector and matrices for Dirichlet conditions
        for key, value in self.mesh.bc_points["dirichlet"].items():
            # Adjust the load vector
            self.F -= (self.A[:, key] + self.M[:, key]).reshape(-1, 1) * value
            # Zero out the rows and columns in the matrices
            self.A[key, :] = 0
            self.A[:, key] = 0
            self.M[key, :] = 0
            self.M[:, key] = 0
            # Set diagonal entry to 1 for the stiffness matrix
            self.A[key, key] = 1
            # Set the corresponding entry in the load vector
            self.F[key] = value

    def process(self):
        """
        Build the global matrices and apply boundary conditions.
        """        
        # Initialize the A and b
        self.A = np.zeros((self.n_points, self.n_points))
        self.F = np.zeros((self.n_points, 1))
        
        # Assemble the global matrices and vector
        self.set_A_M_b()

        # Apply boundary conditions
        self.set_boundary_conditions()

    def solve(self):
        """
        Solve the linear system to find the nodal values of u.
        """
        # Solve the linear system (A + M) u = F
        R = self.A + self.M
        self.u = np.linalg.solve(R, self.F)


class ErrorEstimator:
    """
    A class to compute error estimates using quadrature.
    """

    def __init__(self, u_h, mesh, function_p, function_q, u_exact_func, grad_u_exact_func, quadrature_formula):
        """
        Initialize the ErrorEstimator object.

        Args:
            u_h (ndarray): Finite element solution vector.
            mesh (Mesh): Mesh object.
            function_p (function): Coefficient function p(x, y).
            function_q (function): Coefficient function q(x, y).
            u_exact_func (function): Exact solution u(x, y).
            grad_u_exact_func (function): Gradient of the exact solution.
            quadrature_formula (QuadratureFormula): Quadrature formula object.
        """
        self.u_h = u_h
        self.mesh = mesh
        self.function_p = function_p
        self.function_q = function_q
        self.u_exact = u_exact_func
        self.grad_u_exact = grad_u_exact_func
        self.quadrature_formula = quadrature_formula
        self.element_triangle = ElementTriangle()

    def compute_L2_error(self):
        """
        Compute the L2 norm of the error.

        Returns:
            float: L2 norm of the error.
        """
        L2_error_squared = 0.0
        for simplex in self.mesh.triangulation.simplices:
            node_indices = simplex
            p1, p2, p3 = self.mesh.triangulation.points[node_indices]
            u_h_nodes = self.u_h[node_indices]

            twiceArea = abs(self.element_triangle.twice_Ae(p1, p2, p3))

            # For quadrature over the element
            for weight, (xi, eta) in zip(self.quadrature_formula.weights, self.quadrature_formula.quadrature_points):
                # Get global coordinates at quadrature point
                x_q, y_q = self.element_triangle.get_xy(xi, eta, p1, p2, p3)
                # Compute u_h at quadrature point
                psi_values = [getattr(self.element_triangle, f'psi{i+1}')(xi, eta) for i in range(3)]
                u_h_q = sum(u_h_nodes[i] * psi_values[i] for i in range(3))
                # Compute u_e at quadrature point
                u_e_q = self.u_exact(x_q, y_q)
                # Compute integrand
                integrand = (u_h_q - u_e_q)**2
                # Add to L2_error_squared
                L2_error_squared += weight * integrand * twiceArea
        L2_error = np.sqrt(L2_error_squared)
        return L2_error

    def compute_H1_semi_error(self):
        """
        Compute the H1 semi-norm of the error.

        Returns:
            float: H1 semi-norm of the error.
        """
        H1_semi_error_squared = 0.0
        for simplex in self.mesh.triangulation.simplices:
            node_indices = simplex
            p1, p2, p3 = self.mesh.triangulation.points[node_indices]
            u_h_nodes = self.u_h[node_indices]

            twiceArea = abs(self.element_triangle.twice_Ae(p1, p2, p3))

            # Compute gradients of basis functions
            grad_psi = [
                self.element_triangle.grad_xy_psi1(p1, p2, p3),
                self.element_triangle.grad_xy_psi2(p1, p2, p3),
                self.element_triangle.grad_xy_psi3(p1, p2, p3)
            ]

            # Compute grad_u_h (constant over element)
            grad_u_h = sum(u_h_nodes[i] * grad_psi[i] for i in range(3))

            # For quadrature over the element
            for weight, (xi, eta) in zip(self.quadrature_formula.weights, self.quadrature_formula.quadrature_points):
                # Get global coordinates at quadrature point
                x_q, y_q = self.element_triangle.get_xy(xi, eta, p1, p2, p3)
                # Compute grad_u_e at quadrature point
                grad_u_e_q = self.grad_u_exact(x_q, y_q)
                # Compute grad_error
                grad_error = grad_u_h - grad_u_e_q
                # Compute integrand
                integrand = np.dot(grad_error, grad_error)
                # Add to H1_semi_error_squared
                H1_semi_error_squared += weight * integrand * twiceArea
        H1_semi_error = np.sqrt(H1_semi_error_squared)
        return H1_semi_error

    def compute_energy_error(self):
        """
        Compute the energy norm of the error.

        Returns:
            float: Energy norm of the error.
        """
        energy_error_squared = 0.0
        for simplex in self.mesh.triangulation.simplices:
            node_indices = simplex
            p1, p2, p3 = self.mesh.triangulation.points[node_indices]
            u_h_nodes = self.u_h[node_indices]

            twiceArea = abs(self.element_triangle.twice_Ae(p1, p2, p3))

            # Compute gradients of basis functions
            grad_psi = [
                self.element_triangle.grad_xy_psi1(p1, p2, p3),
                self.element_triangle.grad_xy_psi2(p1, p2, p3),
                self.element_triangle.grad_xy_psi3(p1, p2, p3)
            ]

            # Compute grad_u_h (constant over element)
            grad_u_h = sum(u_h_nodes[i] * grad_psi[i] for i in range(3))

            # For quadrature over the element
            for weight, (xi, eta) in zip(self.quadrature_formula.weights, self.quadrature_formula.quadrature_points):
                # Get global coordinates at quadrature point
                x_q, y_q = self.element_triangle.get_xy(xi, eta, p1, p2, p3)
                # Compute grad_u_e at quadrature point
                grad_u_e_q = self.grad_u_exact(x_q, y_q)
                # Compute grad_error
                grad_error = grad_u_h - grad_u_e_q
                # Evaluate coefficient p(x_q, y_q)
                p_q = self.function_p(x_q, y_q)
                # Compute integrand for gradient part
                integrand_grad = p_q * np.dot(grad_error, grad_error)

                # Compute u_h at quadrature point
                psi_values = [getattr(self.element_triangle, f'psi{i+1}')(xi, eta) for i in range(3)]
                u_h_q = sum(u_h_nodes[i] * psi_values[i] for i in range(3))
                # Compute u_e at quadrature point
                u_e_q = self.u_exact(x_q, y_q)
                u_error_q = u_h_q - u_e_q
                # Evaluate coefficient q(x_q, y_q)
                q_q = self.function_q(x_q, y_q)
                # Compute integrand for u part
                integrand_u = q_q * u_error_q**2

                # Total integrand
                integrand = integrand_grad + integrand_u
                # Add to energy_error_squared
                energy_error_squared += weight * integrand * twiceArea
        energy_error = np.sqrt(energy_error_squared)
        return energy_error


def main():
    """
    Main script to set up and solve the 2D self-adjoint elliptic PDE.
    """
    # Right-hand side function f(x, y)
    def function_f(x, y):
        return -x * y**2 * np.exp(y)

    # Coefficient functions p(x, y) and q(x, y)
    def function_p(x, y):
        return x**2 + y**2   # Coefficient function p(x, y)

    def function_q(x, y):
        return x**2 + 2 * y + 2   # Coefficient function q(x, y)

    # Exact solution and its gradient
    def u_exact(x, y):
        return x * np.exp(y)

    def grad_u_exact(x, y):
        du_dx = np.exp(y)
        du_dy = x * np.exp(y)
        return np.array([du_dx, du_dy])
    
    # Dirichlet boundary condition function
    def uBC(x, y):
        return x * np.exp(y)

    # Domain parameters
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Lists to store mesh sizes and errors
    mesh_sizes = []
    errors_L2 = []
    errors_H1 = []
    errors_energy = []

    previous_L2_error = None
    previous_H1_error = None
    previous_energy_error = None
    previous_h = None

    # Header for the output
    print(f"{'Mesh h':>10} {'Nodes':>10} {'L2 Error':>12} {'Rate L2':>10} {'H1 Error':>12} {'Rate H1':>10} {'Energy Error':>14} {'Rate Energy':>12}")

    # Loop over different mesh refinements
    for n in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
        n_x = n_y = n

        # Compute mesh size h
        h_x = (x_max - x_min) / (n_x - 1)
        h_y = (y_max - y_min) / (n_y - 1)
        h = max(h_x, h_y)
        mesh_sizes.append(h)

        # Create rectangular mesh
        mesh = Mesh(x_min, x_max, n_x, y_min, y_max, n_y)

        # Set Dirichlet boundary conditions at all boundary nodes
        for point_index in mesh.boundary_points:
            x_bc, y_bc = mesh.triangulation.points[point_index]
            mesh.bc_points["dirichlet"][point_index] = uBC(x_bc, y_bc)

        # Solve the 2D self-adjoint elliptic PDE
        # Set up the finite element solver
        solver = FESelfAdjointEllipticPDE2D(mesh, function_p, function_q, function_f)
        # Build matrices and apply boundary conditions
        solver.process()
        # Solve for u
        solver.solve()
        # Get the result
        u_h = solver.u.flatten()

        # Initialize the error estimator
        error_estimator = ErrorEstimator(
            u_h=u_h,
            mesh=mesh,
            function_p=function_p,
            function_q=function_q,
            u_exact_func=u_exact,
            grad_u_exact_func=grad_u_exact,
            quadrature_formula=solver.quadrature_formula
        )

        # Compute errors using the ErrorEstimator
        L2_error = error_estimator.compute_L2_error()
        H1_semi_error = error_estimator.compute_H1_semi_error()
        energy_error = error_estimator.compute_energy_error()

        errors_L2.append(L2_error)
        errors_H1.append(H1_semi_error)
        errors_energy.append(energy_error)

        # Calculate convergence rates if previous errors are available
        if previous_L2_error is not None:
            rate_L2 = np.log(L2_error / previous_L2_error) / np.log(h / previous_h)
            rate_H1 = np.log(H1_semi_error / previous_H1_error) / np.log(h / previous_h)
            rate_energy = np.log(energy_error / previous_energy_error) / np.log(h / previous_h)
        else:
            rate_L2 = rate_H1 = rate_energy = np.nan

        print(f"{h:10.5f} {len(u_h):10d} {L2_error:12.6e} {rate_L2:10.2f} "
              f"{H1_semi_error:12.6e} {rate_H1:10.2f} "
              f"{energy_error:14.6e} {rate_energy:12.2f}")

        # Update previous errors and mesh size
        previous_L2_error = L2_error
        previous_H1_error = H1_semi_error
        previous_energy_error = energy_error
        previous_h = h
        
    # Fit a line to the log-log data to estimate the overall convergence rates
    coeffs_L2 = np.polyfit(np.log(mesh_sizes), np.log(errors_L2), 1)
    rate_L2_overall = coeffs_L2[0]
    coeffs_H1 = np.polyfit(np.log(mesh_sizes), np.log(errors_H1), 1)
    rate_H1_overall = coeffs_H1[0]
    coeffs_energy = np.polyfit(np.log(mesh_sizes), np.log(errors_energy), 1)
    rate_energy_overall = coeffs_energy[0]
    
    # According to Theorem 19, the expected convergence rates are:
    # L2 norm: order 2
    # H1 semi-norm: order 1
    # Energy norm: order 1

    print("\nEstimated overall convergence rates from log-log plot using polynomial fit:")
    print(f"Energy norm: {rate_energy_overall:.2f} (expected 1.0)")
    print(f"H1 semi-norm: {rate_H1_overall:.2f} (expected 1.0)")
    print(f"L2 norm: {rate_L2_overall:.2f} (expected 2.0)")
    

    # Plot the errors versus mesh size h on a log-log plot
    plt.figure(figsize=(10, 8))
    plt.loglog(mesh_sizes, errors_H1, 's-', label=r'$H^1$ Semi-norm Error, Dashed Line Slope = 1, Polynomial Fit Slope = {:.2f}'.format(rate_H1_overall))
    plt.loglog(mesh_sizes, errors_energy, 'h-', label=r'Energy Norm Error, Dashed Line Slope = 1, Polynomial Fit Slope = {:.2f}'.format(rate_energy_overall))
    plt.loglog(mesh_sizes, errors_L2, 'o-', label=r'$L^2$ Error, Dashed Line Slope = 2, Polynomial Fit Slope = {:.2f}'.format(rate_L2_overall))
    
    
    # Add reference lines for expected convergence rates
    h_ref = np.array(mesh_sizes)
    plt.loglog(h_ref, h_ref * errors_energy[0]/h_ref[0], 'k-.')
    plt.loglog(h_ref, h_ref * errors_H1[0]/h_ref[0], 'k-.')
    plt.loglog(h_ref, h_ref**2 * errors_L2[0]/h_ref[0]**2, 'k--')
    

    plt.xlabel('Mesh size h (log scale)')
    plt.ylabel('Error (log scale)')
    plt.title('Error vs. Mesh Size with Convergence Rates (log-log scale)')
    plt.legend()
    
    # Plotting the solution and exact solution for visualization
    # Set specific mesh size for plotting
    n_x = 16
    n_y = 20
    mesh = Mesh(x_min, x_max, n_x, y_min, y_max, n_y)
    # Set Dirichlet boundary conditions at all boundary nodes
    for point_index in mesh.boundary_points:
        x_bc, y_bc = mesh.triangulation.points[point_index]
        mesh.bc_points["dirichlet"][point_index] = uBC(x_bc, y_bc)

    solver = FESelfAdjointEllipticPDE2D(mesh, function_p, function_q, function_f)
    # Build matrices and apply boundary conditions
    solver.process()
    # Solve for u
    solver.solve()
    # Get the result
    u_h = solver.u.flatten()
    # Analytical solution
    x = mesh.triangulation.points[:, 0]
    y = mesh.triangulation.points[:, 1]

    # Plotting the finite element solution and exact solution
    ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u_h, linewidth=0.2, antialiased=True, cmap="tab20b_r", label="Finite Element Solution")
    ax.scatter(x, y, u_exact(x, y), marker='o', c='black', s=1, label="Exact Solution")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$u(x,y)$')
    ax.set_title('Finite Element Solution and Exact Solution')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
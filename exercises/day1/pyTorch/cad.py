"""
Differentiable CAD Algorithms with Bézier Curves using PyTorch

This module implements Bézier curves and radial blade design algorithms
with automatic differentiation capabilities using PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from scipy.integrate import quad
import math


class Curve(ABC):
    """Abstract base class for curves"""
    pass


def radial_transform(zr: torch.Tensor) -> torch.Tensor:
    """
    Conformal transform that transforms rectangular strip to a circle

    Args:
        zr: Input coordinates [r, theta] where r is log-radius and theta is angle

    Returns:
        Cartesian coordinates [x, y]
    """
    if zr.dim() == 1:
        z0r1 = torch.exp(zr[0]) * torch.cos(zr[1])
        z0r2 = torch.exp(zr[0]) * torch.sin(zr[1])
        return torch.stack([z0r1, z0r2])
    else:
        z0r1 = torch.exp(zr[0]) * torch.cos(zr[1])
        z0r2 = torch.exp(zr[0]) * torch.sin(zr[1])
        return torch.stack([z0r1, z0r2], dim=0)


def empty_transform(zr: torch.Tensor) -> torch.Tensor:
    """Identity transformation (no transform)"""
    return zr


class BezierCurve(Curve):
    """
    Bézier curve implementation with PyTorch backend

    Attributes:
        control_points: Control points of the curve [dim x n_points]
        transformation: Function to transform coordinates
    """

    def __init__(self, control_points: torch.Tensor, transformation: Callable = empty_transform):
        """
        Initialize Bézier curve

        Args:
            control_points: Control points tensor of shape [dim, n_points]
            transformation: Optional coordinate transformation function
        """
        self.control_points = control_points.clone()
        self.transformation = transformation
        self.n_points = control_points.shape[1]
        self.dim = control_points.shape[0]

    def de_casteljau(self, u: torch.Tensor) -> torch.Tensor:
        """
        De Casteljau's algorithm to evaluate point on curve

        Args:
            u: Parameter value(s) in [0, 1]

        Returns:
            Point(s) on curve
        """
        # Ensure u is a proper tensor
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(float(u), dtype=torch.float32)

        # Handle scalar input
        if u.dim() == 0:
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Clamp u to valid range
        u = torch.clamp(u, 0.0, 1.0)

        # Initialize working array
        n = self.n_points - 1
        q = self.control_points.clone()

        # De Casteljau iteration
        for k in range(n):
            q_new = torch.zeros_like(q[:, :n-k])
            for i in range(n - k):
                q_new[:, i] = (1.0 - u) * q[:, i] + u * q[:, i + 1]
            q = q_new

        result = self.transformation(q[:, 0])

        if squeeze_output and result.dim() > 1:
            result = result.squeeze(-1)

        return result

    def param_to_point(self, u: torch.Tensor) -> torch.Tensor:
        """Alias for de_casteljau for compatibility"""
        return self.de_casteljau(u)

    def tangent_at_param(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute tangent vector at parameter u using automatic differentiation

        Args:
            u: Parameter value

        Returns:
            Tangent vector
        """
        # Ensure u is a proper scalar tensor with gradients
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(float(u), dtype=torch.float32)

        u = u.clone().detach().requires_grad_(True)

        point = self.de_casteljau(u)

        # For multi-dimensional case (2D/3D)
        gradients = []
        for i in range(point.shape[0]):
            grad = torch.autograd.grad(point[i], u, retain_graph=True,
                                     create_graph=True)[0]
            gradients.append(grad)

        return torch.stack(gradients)

    def unit_tangent_at_param(self, u: torch.Tensor) -> torch.Tensor:
        """Compute unit tangent vector at parameter u"""
        tangent = self.tangent_at_param(u)
        return tangent / torch.norm(tangent)

    def normal_at_param(self, u: torch.Tensor) -> torch.Tensor:
        """Compute normal vector at parameter u (2D only)"""
        tangent = self.tangent_at_param(u)
        # Rotate 90 degrees: [x, y] -> [-y, x]
        return torch.stack([-tangent[1], tangent[0]])

    def unit_normal_at_param(self, u: torch.Tensor) -> torch.Tensor:
        """Compute unit normal vector at parameter u"""
        normal = self.normal_at_param(u)
        return normal / torch.norm(normal)

    def curvature_at_param(self, u: torch.Tensor) -> torch.Tensor:
        """Compute curvature at parameter u using second derivatives"""
        u = u.clone().detach().requires_grad_(True)

        # First derivative
        point = self.de_casteljau(u)
        first_deriv = []
        for i in range(point.shape[0]):
            grad = torch.autograd.grad(point[i], u, create_graph=True, retain_graph=True)[0]
            first_deriv.append(grad)
        first_deriv = torch.stack(first_deriv)

        # Second derivative
        second_deriv = []
        for i in range(first_deriv.shape[0]):
            grad = torch.autograd.grad(first_deriv[i], u, retain_graph=True)[0]
            second_deriv.append(grad)
        second_deriv = torch.stack(second_deriv)

        return second_deriv

    def curve_length_integrand(self, u: float) -> float:
        """
        Integrand for curve length calculation

        Args:
            u: Parameter value

        Returns:
            Speed at parameter u
        """
        u_tensor = torch.tensor(u, dtype=torch.float32)
        try:
            tangent = self.tangent_at_param(u_tensor)
            return float(torch.norm(tangent))
        except:
            # Fallback to finite differences if autograd fails
            h = 1e-6
            u1 = max(0.0, u - h)
            u2 = min(1.0, u + h)
            p1 = self.de_casteljau(torch.tensor(u1, dtype=torch.float32))
            p2 = self.de_casteljau(torch.tensor(u2, dtype=torch.float32))
            diff = p2 - p1
            return float(torch.norm(diff) / (u2 - u1))

    def curve_length(self, u_start: float = 0.0, u_end: float = 1.0) -> float:
        """
        Calculate curve length using numerical integration

        Args:
            u_start: Start parameter
            u_end: End parameter

        Returns:
            Curve length
        """
        try:
            # Try numerical integration first
            length, _ = quad(self.curve_length_integrand, u_start, u_end,
                            epsabs=1e-6, epsrel=1e-6, limit=100)
            # Sanity check: if length is unreasonably large, use discrete approximation
            if length > 10000:  # Threshold for "too large"
                return self._curve_length_discrete(u_start, u_end)
            return length

        except:
            # Fallback to discrete approximation
            return self._curve_length_discrete(u_start, u_end)

    def _curve_length_discrete(self, u_start: float = 0.0, u_end: float = 1.0,
                              n_points: int = 1000) -> float:
        """
        Calculate curve length using discrete approximation

        Args:
            u_start: Start parameter
            u_end: End parameter
            n_points: Number of discretization points

        Returns:
            Approximate curve length
        """
        u_values = torch.linspace(u_start, u_end, n_points, dtype=torch.float32)
        points = []

        for u in u_values:
            point = self.de_casteljau(u)
            points.append(point.detach())

        points = torch.stack(points)

        # Calculate distances between consecutive points
        diffs = points[1:] - points[:-1]
        distances = torch.norm(diffs, dim=1)
        total_length = torch.sum(distances)

        return float(total_length)

    def arc_param_to_param(self, u_arc: float, max_iterations: int = 100,
                          tolerance: float = 1e-10) -> float:
        """
        Convert arc length parameter to curve parameter using Newton's method

        Args:
            u_arc: Arc length parameter (normalized)
            max_iterations: Maximum Newton iterations
            tolerance: Convergence tolerance

        Returns:
            Curve parameter t
        """
        total_length = self.curve_length()

        def M_function(t: float) -> float:
            """Objective function: current_length/total_length - u_arc"""
            current_length = self.curve_length(0.0, t)
            return current_length / total_length - u_arc

        def M_derivative(t: float) -> float:
            """Derivative of M function"""
            return self.curve_length_integrand(t) / total_length

        # Newton's method
        t = u_arc  # Initial guess
        for _ in range(max_iterations):
            f_val = M_function(t)
            if abs(f_val) < tolerance:
                break
            f_prime = M_derivative(t)
            if abs(f_prime) < 1e-15:
                break
            t_new = t - f_val / f_prime
            t_new = max(0.0, min(1.0, t_new))  # Clamp to [0, 1]
            t = t_new

        return t

    def nearest_param_to_point(self, target_point: torch.Tensor,
                              initial_guess: float = 0.5) -> float:
        """
        Find parameter value of point on curve nearest to target point

        Args:
            target_point: Target point
            initial_guess: Initial parameter guess

        Returns:
            Parameter value of nearest point
        """
        u = torch.tensor(initial_guess, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([u], lr=0.01)

        for _ in range(1000):
            optimizer.zero_grad()

            u_clamped = torch.clamp(u, 0.0, 1.0)
            curve_point = self.de_casteljau(u_clamped)
            diff = curve_point - target_point
            loss = torch.sum(diff * diff)

            loss.backward()
            optimizer.step()

            # Clamp u to valid range
            with torch.no_grad():
                u.data = torch.clamp(u.data, 0.0, 1.0)

            if loss.item() < 1e-8:
                break

        return float(u.detach())


class RadialBlade:
    """
    Radial diffuser blade design with Bézier curves

    Attributes:
        n_blades: Number of blades
        inlet_angle: Inlet angle (radians)
        exit_angle: Exit angle (radians)
        stagger_angle: Stagger angle (radians)
        inner_radius: Inner radius
        outer_radius: Outer radius
        leading_edge_radius: Leading edge radius
        trailing_edge_radius: Trailing edge radius
        alpha_ps: Trailing edge slope pressure side
        alpha_ss: Trailing edge slope suction side
        ps_thickness_distribution: Pressure side thickness distribution
        ss_thickness_distribution: Suction side thickness distribution
    """

    def __init__(self, n_blades: int, inlet_angle: float, exit_angle: float,
                 stagger_angle: float, inner_radius: float, outer_radius: float,
                 leading_edge_radius: float, trailing_edge_radius: float,
                 alpha_ps: float, alpha_ss: float,
                 ps_thickness_distribution: BezierCurve,
                 ss_thickness_distribution: BezierCurve):

        self.n_blades = n_blades
        self.inlet_angle = inlet_angle
        self.exit_angle = exit_angle
        self.stagger_angle = stagger_angle
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.leading_edge_radius = leading_edge_radius
        self.trailing_edge_radius = trailing_edge_radius
        self.alpha_ps = alpha_ps
        self.alpha_ss = alpha_ss
        self.ps_thickness_distribution = ps_thickness_distribution
        self.ss_thickness_distribution = ss_thickness_distribution

    def blade_camber_curve(self) -> BezierCurve:
        """Generate the camber geometry from the blade topology"""
        t_out = math.log(self.outer_radius - self.trailing_edge_radius)
        t_in = math.log(self.inner_radius)
        c_ax = t_out - t_in
        c_vx = c_ax * math.tan(self.stagger_angle)

        # Inlet angle line: y = ax + c
        a = math.tan(self.inlet_angle)
        c = -t_in * a

        # Outlet angle line: y = bx + d
        b = -math.tan(self.exit_angle)
        d = -(c_vx + b * t_out)

        # Intersection point
        t1 = (d - c) / (a - b)
        t2 = a * (d - c) / (a - b) + c

        # Quadratic Bézier control points for camber line
        control_points = torch.tensor([
            [t_in, 0.0],
            [t1, t2],
            [t_out, -c_vx]
        ], dtype=torch.float32).T

        return BezierCurve(control_points, radial_transform)

    def curve_from_camber(self, camber: BezierCurve, le_radius: float,
                         te_radius: float, alpha: float, normal_sign: float,
                         thickness: BezierCurve) -> BezierCurve:
        """
        Generate pressure or suction side curve from camber line and thickness distribution
        """
        thickness_points = thickness.control_points
        n_thickness_points = thickness_points.shape[1]

        # Create curve control points with proper size
        n_curve_points = n_thickness_points + 4
        curve_pts = torch.zeros((2, n_curve_points), dtype=torch.float32)

        # First point from camber start
        camber_start_param = camber.control_points[:, 0]  # Get untransformed coordinates
        curve_pts[:, 0] = camber.transformation(camber_start_param)

        # Middle points based on thickness and normal offset from camber
        for i in range(n_thickness_points):
            j = i + 2  # Start from index 2
            if j >= n_curve_points - 2:  # Don't overwrite last two points
                break

            u_param = thickness_points[0, i]
            u_tensor = u_param.clone().detach() if isinstance(u_param, torch.Tensor) else torch.tensor(u_param, dtype=torch.float32)

            # Get camber point and normal
            camber_point = camber.de_casteljau(u_tensor)
            unit_normal = camber.unit_normal_at_param(u_tensor)
            thickness_val = thickness_points[1, i]

            # Apply thickness offset
            curve_pts[:, j] = camber_point - normal_sign * unit_normal * thickness_val

        # Second control point based on leading edge radius
        u_zero = torch.tensor(0.0, dtype=torch.float32)
        unit_normal = camber.unit_normal_at_param(u_zero)

        # Calculate proper offset for leading edge
        if n_thickness_points > 0:
            u_first = thickness_points[0, 0]
            u_first_tensor = u_first.clone().detach() if isinstance(u_first, torch.Tensor) else torch.tensor(u_first, dtype=torch.float32)
            camber_first = camber.de_casteljau(u_first_tensor)

            diff = curve_pts[:, 0] - camber_first
            unit_normal_val = unit_normal.detach()
            b = torch.dot(diff, unit_normal_val)
            b_perp_norm = torch.norm(diff - b * unit_normal_val)

            # Scale factor for leading edge
            scale_factor = torch.sqrt(torch.tensor(le_radius * (n_curve_points - 2.0) / (n_curve_points - 1.0), dtype=torch.float32))
            a = scale_factor * torch.sqrt(b_perp_norm)
        else:
            a = torch.tensor(le_radius, dtype=torch.float32)

        curve_pts[:, 1] = curve_pts[:, 0] - normal_sign * a * unit_normal

        # Last two points for trailing edge
        u_one = torch.tensor(1.0, dtype=torch.float32)
        camber_end = camber.de_casteljau(u_one)
        normal_end = camber.normal_at_param(u_one)
        normal_magnitude = torch.norm(normal_end)

        if normal_magnitude > 1e-10:
            unit_normal_end = normal_end / normal_magnitude

            # Apply trailing edge rotation
            alpha_rad = torch.tensor(alpha * normal_sign, dtype=torch.float32)
            cos_alpha = torch.cos(alpha_rad)
            sin_alpha = torch.sin(alpha_rad)

            # Trailing edge offset
            te_offset = unit_normal_end * te_radius
            rotated_offset = torch.stack([
                te_offset[0] * cos_alpha - te_offset[1] * sin_alpha,
                te_offset[0] * sin_alpha + te_offset[1] * cos_alpha
            ])

            curve_pts[:, -1] = camber_end - normal_sign * rotated_offset

            # Second to last point
            tangent_end = torch.stack([normal_end[1], -normal_end[0]])  # Perpendicular to normal
            tangent_magnitude = torch.norm(tangent_end)

            if tangent_magnitude > 1e-10:
                tangent_end = tangent_end / tangent_magnitude * (normal_magnitude / (n_curve_points - 1.0))

                rotated_tangent = torch.stack([
                    tangent_end[0] * cos_alpha - tangent_end[1] * sin_alpha,
                    tangent_end[0] * sin_alpha + tangent_end[1] * cos_alpha
                ])

                curve_pts[:, -2] = curve_pts[:, -1] - rotated_tangent
            else:
                curve_pts[:, -2] = curve_pts[:, -1]
        else:
            # Fallback if normal computation fails
            curve_pts[:, -1] = camber_end
            curve_pts[:, -2] = camber_end

        return BezierCurve(curve_pts)

    @property
    def dim(self) -> int:
        """Dimension of the curve (2D for this implementation)"""
        return 2

    def pressure_side_curve(self) -> BezierCurve:
        """Generate pressure side curve"""
        camber = self.blade_camber_curve()
        return self.curve_from_camber(
            camber, self.leading_edge_radius, self.trailing_edge_radius,
            self.alpha_ps, 1.0, self.ps_thickness_distribution
        )

    def suction_side_curve(self) -> BezierCurve:
        """Generate suction side curve"""
        camber = self.blade_camber_curve()
        return self.curve_from_camber(
            camber, self.leading_edge_radius, self.trailing_edge_radius,
            self.alpha_ss, -1.0, self.ss_thickness_distribution
        )


def write_blade_geometry_gmsh(blade: RadialBlade, filename: str):
    """
    Write blade geometry to GMSH file format

    Args:
        blade: RadialBlade instance
        filename: Output filename
    """
    camber = blade.blade_camber_curve()
    ps = blade.pressure_side_curve()
    ss = blade.suction_side_curve()

    with open(filename, 'w') as f:
        f.write('SetFactory("OpenCASCADE");\n\n')

        # Write pressure side control points
        point_id = 1
        ps_points = ps.control_points.detach().numpy()
        for i in range(ps_points.shape[1]):
            f.write(f'Point({point_id}) = {{{ps_points[0, i]:.6f}, {ps_points[1, i]:.6f}, 0.0}}; // PS {i+1}\n')
            point_id += 1

        f.write('\n')

        # Write TE circle center
        camber_end = camber.transformation(camber.control_points[:, -1]).detach().numpy()
        f.write(f'Point({point_id}) = {{{camber_end[0]:.6f}, {camber_end[1]:.6f}, 0.0}}; // TE circle\n')
        te_center_id = point_id
        point_id += 1

        f.write('\n')

        # Write suction side control points (reverse order, skip first)
        ss_points = ss.control_points.detach().numpy()
        ss_start_id = point_id
        for i in range(ss_points.shape[1] - 1, 0, -1):
            f.write(f'Point({point_id}) = {{{ss_points[0, i]:.6f}, {ss_points[1, i]:.6f}, 0.0}}; // SS {i+1}\n')
            point_id += 1

        f.write('\n')

        # PS Bézier curve
        f.write('Bezier(1) = { 1')
        for i in range(2, ps_points.shape[1] + 1):
            f.write(f', {i}')
        f.write(' };\n')

        # TE circle
        ps_end_id = ps_points.shape[1]
        f.write(f'Circle(2) = {{{ps_end_id}, {te_center_id}, {ss_start_id}}}; // TE Circle\n\n')

        # SS Bézier curve
        f.write(f'Bezier(3) = {{{ss_start_id}')
        for i in range(1, ss_points.shape[1] - 1):
            f.write(f', {ss_start_id + i}')
        f.write(', 1};\n\n')

        # Create surfaces and blade repetition
        f.write('Curve Loop(1) = {3, 1, 2};\n')
        f.write('Plane Surface(1) = {1};\n')
        f.write(f'Circle(4) = {{0, 0, 0, {blade.inner_radius * 0.9:.6f}, 0, 2*Pi}};\n')
        f.write(f'Circle(5) = {{0, 0, 0, {blade.outer_radius * 1.1:.6f}, 0, 2*Pi}};\n')
        f.write('Curve Loop(2) = {5};\n')
        f.write('Curve Loop(3) = {4};\n')
        f.write('Plane Surface(2) = {2, 3};\n')

        # Replicate blades
        for i in range(1, blade.n_blades):
            angle = 2.0 * math.pi * i / blade.n_blades
            f.write(f'Rotate{{ {{0, 0, 1}}, {{0, 0, 0}}, {angle:.6f} }} {{\n')
            f.write('  Duplicata { Surface{1}; }\n')
            f.write('}\n')

        # Boolean difference
        f.write('BooleanDifference{ Surface{2}; Delete; }{ Surface{1')
        for i in range(1, blade.n_blades):
            f.write(f', {i + 2}')
        f.write('}; Delete; }\n')


def create_demo_blade() -> RadialBlade:
    """Create a demo radial blade for testing"""
    deg2rad = math.pi / 180.0

    # Thickness distributions
    thickness_ps_points = torch.tensor([[0.6044527622], [2.0000013924]], dtype=torch.float32)
    thickness_ss_points = torch.tensor([[0.65037604627], [1.9439006187]], dtype=torch.float32)

    thickness_ps = BezierCurve(thickness_ps_points)
    thickness_ss = BezierCurve(thickness_ss_points)

    blade = RadialBlade(
        n_blades=15,
        inlet_angle=112.0 * deg2rad,
        exit_angle=56.0 * deg2rad,
        stagger_angle=62.76 * deg2rad,
        inner_radius=80.0,
        outer_radius=112.0,
        leading_edge_radius=1.0,
        trailing_edge_radius=0.5,
        alpha_ps=5.0 * deg2rad,
        alpha_ss=5.0 * deg2rad,
        ps_thickness_distribution=thickness_ps,
        ss_thickness_distribution=thickness_ss
    )

    return blade


def plot_blade_geometry(blade: RadialBlade, n_points: int = 100):
    """Plot blade geometry"""
    camber = blade.blade_camber_curve()
    ps = blade.pressure_side_curve()
    ss = blade.suction_side_curve()

    # Generate points for plotting
    u_values = torch.linspace(0, 1, n_points)

    # Camber line points
    camber_points = []
    for u in u_values:
        point = camber.de_casteljau(u)
        camber_points.append(point.detach().numpy())
    camber_points = np.array(camber_points)

    # Control points for visualization
    ps_control = ps.control_points.detach().numpy()
    ss_control = ss.control_points.detach().numpy()

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(camber_points[:, 0], camber_points[:, 1], 'b-', label='Camber Line', linewidth=2)
    plt.scatter(ps_control[0, :], ps_control[1, :], c='red', s=50, label='PS Control Points', zorder=5)
    plt.scatter(ss_control[0, :], ss_control[1, :], c='green', s=50, label='SS Control Points', zorder=5)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Radial Diffuser Blade Geometry')
    plt.xlabel('X')
    plt.ylabel('Y')

    return plt.gcf()


if __name__ == "__main__":
    # Create and test demo blade
    print("Creating demo radial blade...")
    blade = create_demo_blade()

    # Calculate curve lengths
    print("\nCalculating curve lengths...")
    ps = blade.pressure_side_curve()
    ss = blade.suction_side_curve()

    ps_length = ps.curve_length()
    ss_length = ss.curve_length()

    print(f"Pressure side curve length: {ps_length:.6f}")
    print(f"Suction side curve length: {ss_length:.6f}")

    # Test automatic differentiation
    print("\nTesting automatic differentiation...")
    u_test = torch.tensor(0.5, dtype=torch.float32)

    camber = blade.blade_camber_curve()
    point = camber.de_casteljau(u_test)
    tangent = camber.tangent_at_param(u_test)
    normal = camber.normal_at_param(u_test)

    print(f"Point at u=0.5: {point}")
    print(f"Tangent at u=0.5: {tangent}")
    print(f"Normal at u=0.5: {normal}")

    # Generate GMSH file
    print("\nGenerating GMSH geometry file...")
    write_blade_geometry_gmsh(blade, 'blade.geo')

    # Create plot
    print("Creating plot...")
    fig = plot_blade_geometry(blade)
    plt.savefig('blade_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Demo completed successfully!")

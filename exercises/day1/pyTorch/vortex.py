"""
Exact solution to Isentropic Vortex Convection - PyTorch Implementation

This module provides PyTorch implementations of the analytical solutions for
the 2D Isentropic vortex, including velocity perturbations and vorticity
calculations using both complex-step differentiation and analytical methods.

Original Julia implementation ported to PyTorch.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

class IsentropicVortex:
    """
    Class for computing isentropic vortex solutions and visualizations.

    For a vortex of strength Γ centered at (x₀, y₀) with characteristic radius R,
    the perturbation velocities are given by:

    (u', v') = (Γ/(2πR²)) * exp[(1 - (r/R)²)/2] * (y₀ - y, x - x₀)

    where r² = (x-x₀)² + (y-y₀)²
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the IsentropicVortex class.

        Args:
            device: PyTorch device ('cpu', 'cuda', etc.). If None, auto-detects.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def velocity_perturbations(self, x: torch.Tensor, y: torch.Tensor,
                             gamma: float, x0: float, y0: float,
                             R: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate velocity perturbations (u', v') for the isentropic vortex.

        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            gamma: Vortex strength (Γ)
            x0: Vortex center x-coordinate
            y0: Vortex center y-coordinate
            R: Characteristic radius

        Returns:
            Tuple of (u_prime, v_prime) velocity perturbations
        """
        # Ensure tensors are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)

        # Calculate distances from vortex center
        dx = x - x0
        dy = y - y0
        d2 = dx**2 + dy**2
        R2 = R**2

        # Calculate the exponential term
        term = torch.exp(-d2 / R2 / 2)

        # Calculate velocity perturbations
        u_prime = -gamma / (2 * np.pi) * dy / R2 * term
        v_prime = gamma / (2 * np.pi) * dx / R2 * term

        return u_prime, v_prime

    def velocity_magnitude(self, x: torch.Tensor, y: torch.Tensor,
                          gamma: float, x0: float, y0: float,
                          R: float) -> torch.Tensor:
        """
        Calculate velocity magnitude for the isentropic vortex.

        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            gamma: Vortex strength (Γ)
            x0: Vortex center x-coordinate
            y0: Vortex center y-coordinate
            R: Characteristic radius

        Returns:
            Velocity magnitude tensor
        """
        u_prime, v_prime = self.velocity_perturbations(x, y, gamma, x0, y0, R)
        return torch.sqrt(u_prime**2 + v_prime**2)

    def vorticity_complex_step(self, x: torch.Tensor, y: torch.Tensor,
                              gamma: float, x0: float, y0: float,
                              R: float, epsilon: float = 1e-12) -> torch.Tensor:
        """
        Calculate vorticity using complex-step differentiation.

        Vorticity ω = ∂v'/∂x - ∂u'/∂y

        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            gamma: Vortex strength (Γ)
            x0: Vortex center x-coordinate
            y0: Vortex center y-coordinate
            R: Characteristic radius
            epsilon: Complex step size

        Returns:
            Vorticity tensor
        """
        # Convert to complex tensors
        x_complex = x.to(torch.complex64)
        y_complex = y.to(torch.complex64)

        # Calculate dv/dx using complex step
        x_perturbed = x_complex + epsilon * 1j
        _, v_prime_x = self.velocity_perturbations(x_perturbed, y_complex, gamma, x0, y0, R)
        dvdx = torch.imag(v_prime_x) / epsilon

        # Calculate du/dy using complex step
        y_perturbed = y_complex + epsilon * 1j
        u_prime_y, _ = self.velocity_perturbations(x_complex, y_perturbed, gamma, x0, y0, R)
        dudy = torch.imag(u_prime_y) / epsilon

        # Vorticity = dv/dx - du/dy
        vorticity = dvdx - dudy

        return torch.real(vorticity)

    def vorticity_analytical(self, x: torch.Tensor, y: torch.Tensor,
                           gamma: float, x0: float, y0: float,
                           R: float) -> torch.Tensor:
        """
        Calculate vorticity using the analytical expression.

        From the analytical derivation:
        ω = (Γ/(2πR²)) * [2 - (r/R)²] * exp[(1 - (r/R)²)/2]

        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            gamma: Vortex strength (Γ)
            x0: Vortex center x-coordinate
            y0: Vortex center y-coordinate
            R: Characteristic radius

        Returns:
            Vorticity tensor
        """
        # Ensure tensors are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)

        # Calculate distances from vortex center
        dx = x - x0
        dy = y - y0
        r2 = dx**2 + dy**2
        R2 = R**2

        # Analytical vorticity expression
        #term1 = gamma / (2 * np.pi * R2)
        #term2 = 2 - r2 / R2
        #term3 = torch.exp((1 - r2 / R2) / 2)

        vorticity = 0.0#term1 * term2 * term3

        return vorticity

    def temperature_perturbation(self, x: torch.Tensor, y: torch.Tensor,
                                gamma: float, x0: float, y0: float,
                                R: float, gamma_gas: float = 1.4) -> torch.Tensor:
        """
        Calculate temperature perturbation for the isentropic vortex.

        T∞ = -((γ-1)/(2γ)) * (Γ/(2πR²))² * exp[1 - (r/R)²]

        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            gamma: Vortex strength (Γ)
            x0: Vortex center x-coordinate
            y0: Vortex center y-coordinate
            R: Characteristic radius
            gamma_gas: Specific heat ratio (default: 1.4 for air)

        Returns:
            Temperature perturbation tensor
        """
        # Ensure tensors are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)

        # Calculate distances from vortex center
        dx = x - x0
        dy = y - y0
        r2 = dx**2 + dy**2
        R2 = R**2

        # Temperature perturbation
        factor = -(gamma_gas - 1) / (2 * gamma_gas)
        circulation_term = (gamma / (2 * np.pi * R2))**2
        exponential_term = torch.exp(1 - r2 / R2)

        T_perturbation = factor * circulation_term * exponential_term

        return T_perturbation

def create_grid(x_range: Tuple[float, float], y_range: Tuple[float, float],
                num_points: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 2D grid of points for evaluation.

    Args:
        x_range: (x_min, x_max) range
        y_range: (y_min, y_max) range
        num_points: Number of points in each direction
        device: PyTorch device

    Returns:
        Tuple of (X, Y) coordinate meshgrids
    """
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    y = torch.linspace(y_range[0], y_range[1], num_points, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y

def plot_contours(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor,
                 title: str, xlabel: str = 'x coordinate',
                 ylabel: str = 'y coordinate', figsize: Tuple[int, int] = (8, 6)):
    """
    Plot contour plots of the given data.

    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Values to contour
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Convert to numpy for plotting
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    Z_np = Z.cpu().numpy()

    # Create contour plot
    contour = plt.contour(X_np, Y_np, Z_np, levels=20)
    plt.contourf(X_np, Y_np, Z_np, levels=20, alpha=0.6, cmap='viridis')
    plt.colorbar(label=title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def main():
    """
    Main function demonstrating the isentropic vortex calculations and plots.
    """
    print("Isentropic Vortex Convection - PyTorch Implementation")
    print("=" * 55)

    # Initialize the vortex class
    vortex = IsentropicVortex()
    print(f"Using device: {vortex.device}")

    # Vortex parameters
    gamma = 1.0          # Vortex strength (Γ)
    x0, y0 = 0.0, 0.0   # Vortex center
    R = 0.1             # Characteristic radius
    num_points = 100    # Grid resolution

    print(f"\nVortex Parameters:")
    print(f"  Vortex Strength (Γ)      = {gamma}")
    print(f"  Vortex center (x0, y0)   = ({x0}, {y0})")
    print(f"  Characteristic Radius (R) = {R}")
    print(f"  Grid resolution          = {num_points}×{num_points}")

    # Create computational grid
    X, Y = create_grid((-1, 1), (-1, 1), num_points, vortex.device)

    # Calculate velocity magnitude
    print("\n1. Computing velocity magnitude...")
    vel_mag = vortex.velocity_magnitude(X, Y, gamma, x0, y0, R)

    # Calculate vorticity using complex-step method
    print("2. Computing vorticity (complex-step method)...")
    vorticity_cs = vortex.vorticity_complex_step(X, Y, gamma, x0, y0, R)

    # Calculate vorticity using analytical method
    #print("3. Computing vorticity (analytical method)...")
    #vorticity_analytical = vortex.vorticity_analytical(X, Y, gamma, x0, y0, R)

    # Compare complex-step and analytical vorticity
    #error = torch.abs(vorticity_cs - vorticity_analytical)
    #max_error = torch.max(error)
    #mean_error = torch.mean(error)
    #print(f"\nVorticity Calculation Comparison:")
    #print(f"  Maximum error between methods: {max_error:.2e}")
    #print(f"  Mean error between methods:    {mean_error:.2e}")

    # Create plots
    print("\nCreating visualizations...")

    # Plot velocity magnitude contours
    plot_contours(X, Y, vel_mag, 'Velocity Magnitude')
    plt.savefig('velocity_magnitude.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot vorticity (complex-step method)
    plot_contours(X, Y, vorticity_cs, 'Vorticity (Complex-Step Method)')
    plt.savefig('vorticity_complex_step.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot vorticity (analytical method)
    #plot_contours(X, Y, vorticity_analytical, 'Vorticity (Analytical Method)')
    #plt.savefig('vorticity_analytical.png', dpi=150, bbox_inches='tight')
    #plt.show()

    # Plot error between methods
    #plot_contours(X, Y, torch.log10(error + 1e-16), 'Log₁₀(Error) Between Methods')
    #plt.savefig('vorticity_error.png', dpi=150, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    main()


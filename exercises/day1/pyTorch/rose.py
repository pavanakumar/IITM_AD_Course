#!/usr/bin/env python3
"""
Rosenbrock Function Optimization: Gradient vs Gradient-Free Methods

This script demonstrates:
1. 3D visualization of the Rosenbrock function
2. Gradient-free optimization (CMAES)
3. Gradient-based optimization (LBFGS with automatic differentiation)
4. Finite difference gradient estimation analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import minimize, differential_evolution
from cma import CMAEvolutionStrategy
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the Rosenbrock function
def rosenbrock_numpy(x, p):
    """Rosenbrock function for numpy arrays (for scipy optimization)"""
    return (p[0] - x[0])**2 + p[1] * (x[1] - x[0]**2)**2

def rosenbrock_torch(x, p):
    """Rosenbrock function for PyTorch tensors"""
    return (p[0] - x[0])**2 + p[1] * (x[1] - x[0]**2)**2

def rosenbrock_for_plot(x, y, p):
    """Rosenbrock function for plotting (numpy arrays)"""
    return (p[0] - x)**2 + p[1] * (y - x**2)**2

# Parameters for Rosenbrock function
param = [1.0, 100.0]

print("="*60)
print("ROSENBROCK FUNCTION OPTIMIZATION COMPARISON")
print("="*60)

# 1. Create 3D surface plot
print("\n1. Creating 3D surface plot...")
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock_for_plot(X, Y, param)

fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1, x2)')
ax1.set_title('Rosenbrock Function')
ax1.view_init(elev=30, azim=40)
fig1.colorbar(surf)
plt.tight_layout()
plt.show(block=False)

print("\n2. OPTIMIZATION USING GRADIENT-FREE METHOD (CMA-ES)")
print("-" * 50)

# Gradient-free optimization using CMA-ES
x0_numpy = np.zeros(2)

def objective_cmaes(x):
    """Objective function for CMA-ES"""
    return rosenbrock_numpy(x, param)

# CMA-ES optimization
start_time = time.time()
es = CMAEvolutionStrategy(x0_numpy, 0.5, {'bounds': [[-2, -2], [2, 2]], 'verbose': -1})
es.optimize(objective_cmaes, maxfun=10000)
cmaes_time = time.time() - start_time

cmaes_result = es.result
print(f"Total number of function evaluations: {cmaes_result.evaluations}")
print(f"Time taken: {cmaes_time:.6f} s")
print(f"x[1] = {cmaes_result.xbest[0]:.6f}")
print(f"x[2] = {cmaes_result.xbest[1]:.6f}")
print(f"Final function value: {cmaes_result.fbest:.8f}")

print("\n3. OPTIMIZATION USING GRADIENT-BASED METHOD (LBFGS)")
print("-" * 50)

# Gradient-based optimization using PyTorch
x0_torch = torch.zeros(2, device=device, requires_grad=True)

# PyTorch LBFGS optimizer
def closure():
    optimizer.zero_grad()
    loss = rosenbrock_torch(x, torch.tensor(param, device=device))
    loss.backward()
    return loss

x = torch.tensor([0.0, 0.0], device=device, requires_grad=True)
optimizer = torch.optim.LBFGS([x], max_iter=100, tolerance_grad=1e-3, tolerance_change=1e-9)

start_time = time.time()
initial_loss = None
num_function_evals = 0

for i in range(100):  # Max iterations
    def closure_with_counter():
        global num_function_evals
        num_function_evals += 1
        optimizer.zero_grad()
        loss = rosenbrock_torch(x, torch.tensor(param, device=device))
        if initial_loss is None:
            globals()['initial_loss'] = loss.item()
        loss.backward()
        return loss
    
    loss = optimizer.step(closure_with_counter)
    
    if torch.norm(x.grad) < 1e-3:  # Convergence check
        break

lbfgs_time = time.time() - start_time

print(f"Total number of function evaluations: {num_function_evals}")
print(f"Time taken: {lbfgs_time:.6f} s")
print(f"x[1] = {x[0].item():.6f}")
print(f"x[2] = {x[1].item():.6f}")
print(f"Final function value: {rosenbrock_torch(x, torch.tensor(param, device=device)).item():.8f}")

print("\n4. FINITE DIFFERENCE GRADIENT ESTIMATION ANALYSIS")
print("-" * 50)
print("\nForward-difference approximation")
print("Analyzing gradient estimation error vs step size...")

# Evaluation point
eval_point = [0.5, 0.5]

# Calculate exact gradient using automatic differentiation
x_exact = torch.tensor(eval_point, requires_grad=True)
f_exact = rosenbrock_torch(x_exact, torch.tensor(param))
f_exact.backward()
exact_grad = x_exact.grad[0].item()  # Gradient w.r.t. first variable

print(f"Exact gradient (w.r.t. x1): {exact_grad:.6f}")

# Forward difference approximation
h_values = np.logspace(-14, -2, 20000)
gradient_errors = []

for h in h_values:
    # Forward difference: (f(x + h*e1) - f(x)) / h
    x_plus_h = [eval_point[0] + h, eval_point[1]]
    f_plus_h = rosenbrock_numpy(x_plus_h, param)
    f_current = rosenbrock_numpy(eval_point, param)
    
    fd_grad = (f_plus_h - f_current) / h
    error = abs(exact_grad - fd_grad)
    gradient_errors.append(error)

# Plot gradient error vs step size
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)
ax2.loglog(h_values, gradient_errors, 'b-', linewidth=1)
ax2.set_xlabel('Step size (h)')
ax2.set_ylabel('Gradient error (AD - FD)')
ax2.set_title('Forward Difference Gradient Error vs Step Size')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n5. CENTRAL DIFFERENCE APPROXIMATION")
print("-" * 50)

# Central difference approximation
#gradient_errors_central = []

#for h in h_values:
    # Central difference: (f(x + h*e1) - f(x - h*e1)) / (2*h)
    #    x_plus_h = [eval_point[0] + h, eval_point[1]]
    #    x_minus_h = [eval_point[0] - h, eval_point[1]]
    
    #    f_plus_h = rosenbrock_numpy(x_plus_h, param)
    #    f_minus_h = rosenbrock_numpy(x_minus_h, param)
    
    #    cd_grad = (f_plus_h - f_minus_h) / (2 * h)
    #    error = abs(exact_grad - cd_grad)
    #    gradient_errors_central.append(error)

# Plot comparison of forward vs central difference
#plt.figure(figsize=(12, 6))
#plt.loglog(h_values, gradient_errors, 'b-', linewidth=1, label='Forward Difference')
#plt.loglog(h_values, gradient_errors_central, 'r-', linewidth=1, label='Central Difference')
#plt.xlabel('Step size (h)')
#plt.ylabel('Gradient error (AD - FD/CD)')
#plt.title('Gradient Error Comparison: Forward vs Central Difference')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.tight_layout()
#plt.show()

#print("\nObservations:")
#print("- Central difference generally provides better accuracy than forward difference")
#print("- Both methods show optimal step sizes around 1e-8 to 1e-6")
#print("- Too small step sizes lead to numerical precision errors")
#print("- Too large step sizes lead to discretization errors")

print("\n6. OPTIMIZATION COMPARISON SUMMARY")
print("-" * 50)
print(f"{'Method':<20} {'Time (s)':<12} {'Function Evals':<15} {'Final x1':<12} {'Final x2':<12}")
print("-" * 70)
print(f"{'CMA-ES':<20} {cmaes_time:<12.6f} {cmaes_result.evaluations:<15} {cmaes_result.xbest[0]:<12.6f} {cmaes_result.xbest[1]:<12.6f}")
print(f"{'LBFGS (PyTorch)':<20} {lbfgs_time:<12.6f} {num_function_evals:<15} {x[0].item():<12.6f} {x[1].item():<12.6f}")

print("\nKey Insights:")
print("1. Gradient-based methods (LBFGS) typically converge faster and with fewer function evaluations")
print("2. Gradient-free methods (CMA-ES) are more robust but require more function evaluations")
print("3. Automatic differentiation provides exact gradients, superior to finite differences")
print("4. PyTorch's autograd makes gradient-based optimization straightforward and efficient")

# Additional PyTorch-specific demonstrations
print("\n7. PYTORCH-SPECIFIC FEATURES")
print("-" * 50)

# Demonstrate gradient computation with different torch functions
print("Demonstrating PyTorch gradient computation capabilities:")

# Create a more complex version using torch functions
def rosenbrock_torch_vectorized(x_tensor):
    """Vectorized Rosenbrock function for batch processing"""
    a, b = 1.0, 100.0
    return (a - x_tensor[..., 0])**2 + b * (x_tensor[..., 1] - x_tensor[..., 0]**2)**2

# Example with batch processing
batch_size = 5
x_batch = torch.randn(batch_size, 2, requires_grad=True)
f_batch = rosenbrock_torch_vectorized(x_batch)
loss_batch = f_batch.mean()
loss_batch.backward()

print(f"Batch gradient computation successful for {batch_size} samples")
print(f"Gradient shape: {x_batch.grad.shape}")
print(f"Mean gradient magnitude: {torch.norm(x_batch.grad, dim=1).mean().item():.6f}")

print("\nScript execution completed successfully!")


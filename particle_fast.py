import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.physics.hydrogen import Psi_nlm
from sympy.abc import r, phi, theta
from sympy import lambdify
import numexpr as ne  # For faster numerical evaluation

# Define quantum numbers for the hydrogen orbital
n = 3  # Principal quantum number
l = 2  # Azimuthal quantum number (0 <= l < n)
m = 0  # Magnetic quantum number (-l <= m <= l)

# Lambdify the symbolic wave function
psi_sym = Psi_nlm(n, l, m, r, phi, theta)
psi_func = lambdify((r, phi, theta), psi_sym, 'numpy')

# Set box size (in Bohr radii); scaled for orbital size
L = 15.0 * n  # Reduced from 30*n for efficiency
max_radius = 10 * n  # Reduced radius to focus on high-probability regions

# Precompute a coarse grid to estimate max probability density
grid_size = 50  # Smaller grid for faster initial estimation
x = np.linspace(-L, L, grid_size)
y = np.linspace(-L, L, grid_size)
z = np.linspace(-L, L, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
R = np.where(R == 0, 1e-10, R)  # Avoid division by zero
Theta = np.arccos(Z / R)
Phi = np.arctan2(Y, X)

# Vectorized computation of probability density
psi = psi_func(R, Phi, Theta)
prob_densities = ne.evaluate("abs(psi)**2")  # Faster with numexpr
max_prob = np.max(prob_densities) * 1.1  # Slight overestimate

# Monte Carlo sampling with vectorized operations
num_points = 5000  # Reduced from 10000 for faster plotting
points = []
batch_size = 10000  # Process points in batches
attempts = 0
max_attempts = 10000000

while len(points) < num_points and attempts < max_attempts:
    # Generate batch of points
    x_batch = np.random.uniform(-L, L, batch_size)
    y_batch = np.random.uniform(-L, L, batch_size)
    z_batch = np.random.uniform(-L, L, batch_size)
    r2 = x_batch**2 + y_batch**2 + z_batch**2
    mask = r2 < max_radius**2
    x_batch, y_batch, z_batch = x_batch[mask], y_batch[mask], z_batch[mask]
    
    if len(x_batch) == 0:
        attempts += len(mask)
        continue
    
    # Compute probability for batch
    R_batch = np.sqrt(r2[mask])
    R_batch = np.where(R_batch == 0, 1e-10, R_batch)
    Theta_batch = np.arccos(z_batch / R_batch)
    Phi_batch = np.arctan2(y_batch, x_batch)
    psi_batch = psi_func(R_batch, Phi_batch, Theta_batch)
    prob_batch = ne.evaluate("abs(psi_batch)**2")
    
    # Acceptance-rejection sampling
    accept = np.random.random(len(prob_batch)) < (prob_batch / max_prob)
    accepted_points = np.vstack((x_batch[accept], y_batch[accept], z_batch[accept])).T
    points.extend(accepted_points)
    attempts += len(mask)

points = np.array(points[:num_points])  # Trim to desired number

# Visualize the point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5)

# Set labels and title
ax.set_xlabel('X (Bohr radii)')
ax.set_ylabel('Y (Bohr radii)')
ax.set_zlabel('Z (Bohr radii)')
ax.set_title(f'Hydrogen Orbital (n={n}, l={l}, m={m})\nOptimized Monte Carlo Sampling')
ax.set_xlim([-L/2, L/2])
ax.set_ylim([-L/2, L/2])
ax.set_zlim([-L/2, L/2])

plt.show()
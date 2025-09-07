import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.physics.hydrogen import Psi_nlm
from sympy.abc import r, phi, theta
from sympy import lambdify

# Define the quantum numbers for the hydrogen orbital (change these for different orbitals)
n = 3  # Principal quantum number
l = 2  # Azimuthal quantum number (0 <= l < n)
m = 0  # Magnetic quantum number (-l <= m <= l)

# Lambdify the symbolic wave function for numerical evaluation
psi_sym = Psi_nlm(n, l, m, r, phi, theta)
psi_func = lambdify((r, phi, theta), psi_sym, 'numpy')

# Function to compute the wave function value at Cartesian coordinates
def compute_psi(x, y, z):
    R = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero at origin
    R = np.where(R == 0, 1e-10, R)
    Theta = np.arccos(z / R)
    Phi = np.arctan2(y, x)
    return psi_func(R, Phi, Theta)

# Set the box size for sampling (in Bohr radii); adjust based on n for larger orbitals
L = 30.0 * n  # Larger for higher n to capture the orbital extent

# First, estimate the maximum probability density using a coarse sampling
num_coarse_samples = 100000
x_coarse = np.random.uniform(-L, L, num_coarse_samples)
y_coarse = np.random.uniform(-L, L, num_coarse_samples)
z_coarse = np.random.uniform(-L, L, num_coarse_samples)
# Filter points within a reasonable radius to improve efficiency
max_radius = 20 * n
inside = x_coarse**2 + y_coarse**2 + z_coarse**2 < max_radius**2
x_coarse, y_coarse, z_coarse = x_coarse[inside], y_coarse[inside], z_coarse[inside]
psi_values = compute_psi(x_coarse, y_coarse, z_coarse)
prob_densities = np.abs(psi_values)**2
max_prob = np.max(prob_densities) * 1.1  # Slight overestimate for safety

# Now, perform rejection sampling to generate points distributed according to |psi|^2
num_points = 10000  # Number of points in the cloud; increase for denser visualization
points = []
attempts = 0
max_attempts = 100000000  # Safety limit to prevent infinite loop

while len(points) < num_points and attempts < max_attempts:
    attempts += 1
    x, y, z = np.random.uniform(-L, L, 3)
    if x**2 + y**2 + z**2 > max_radius**2:
        continue  # Skip points far outside the orbital
    prob = np.abs(compute_psi(x, y, z))**2
    if np.random.random() < (prob / max_prob):
        points.append([x, y, z])

points = np.array(points)

# Visualize the point cloud
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5)

# Set labels and title
ax.set_xlabel('X (Bohr radii)')
ax.set_ylabel('Y (Bohr radii)')
ax.set_zlabel('Z (Bohr radii)')
ax.set_title(f'Hydrogen Orbital Point Cloud (n={n}, l={l}, m={m})\nMonte Carlo Sampling of Electron Probability Density')

# Adjust limits if needed
ax.set_xlim([-L/2, L/2])
ax.set_ylim([-L/2, L/2])
ax.set_zlim([-L/2, L/2])

plt.show()
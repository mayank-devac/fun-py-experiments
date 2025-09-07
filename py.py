import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
wavelength = 1.0  # Wavelength of the light
slit_separation = 5.0  # Distance between the two slits (in units of wavelength)
grid_size_x = 300  # Resolution for x-axis
grid_size_y = 300  # Resolution for y-axis
x_min, x_max = -50, 50  # x-range for the plot
y_min, y_max = 0, 100  # y-range (propagation direction)

# Slit positions (at y=0)
slit1_pos = (-slit_separation / 2, 0)
slit2_pos = (slit_separation / 2, 0)

# Wave number k = 2π / λ
k = 2 * np.pi / wavelength

# Angular frequency ω (assuming speed of light c=1 for simplicity)
omega = k  # Since ω = k * c and c=1

# Create grid
x = np.linspace(x_min, x_max, grid_size_x)
y = np.linspace(y_min, y_max, grid_size_y)
X, Y = np.meshgrid(x, y)

# Function to compute the wave field at time t
def wave_field(X, Y, t):
    r1 = np.sqrt((X - slit1_pos[0])**2 + (Y - slit1_pos[1])**2)
    r2 = np.sqrt((X - slit2_pos[0])**2 + (Y - slit2_pos[1])**2)
    wave1 = np.sin(k * r1 - omega * t)
    wave2 = np.sin(k * r2 - omega * t)
    return wave1 + wave2

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Update function for animation
def update(t):
    field = wave_field(X, Y, t)
    ax.clear()
    pcm = ax.pcolormesh(X, Y, field, cmap='RdBu', shading='auto', vmin=-2, vmax=2)
    ax.set_title(f'Double Slit Wave Interference (t = {t:.2f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y (propagation direction)')
    ax.set_aspect('equal')
    # Optional: Mark slit positions
    ax.plot(slit1_pos[0], slit1_pos[1], 'ko', markersize=5)
    ax.plot(slit2_pos[0], slit2_pos[1], 'ko', markersize=5)
    # Optional: Show intensity pattern on a "screen" at max y
    screen_y = y_max
    screen_indices = np.argmin(np.abs(y - screen_y))
    intensity = field[screen_indices, :]**2
    ax.plot(x, np.ones_like(x) * screen_y, 'k-', linewidth=1)  # Screen line
    ax.plot(x, screen_y + 5 * (intensity / np.max(intensity)), 'g-', linewidth=1)  # Scaled intensity above screen
    return pcm,

# Create animation
anim = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi / omega, 50), blit=False, interval=50)

# Save as GIF
writer = PillowWriter(fps=20)
anim.save('double_slit_experiment.gif', writer=writer)

plt.close(fig)  # Close the figure to prevent static display
print("GIF saved as 'double_slit_experiment.gif'")
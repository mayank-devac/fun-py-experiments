"""
Young's Double Slit Experiment Simulation

This script simulates Young's double-slit experiment, a fundamental demonstration of wave interference.  
Monochromatic light is passed through two narrow slits, creating an interference pattern on a screen.
The simulation visualizes the intensity distribution of the light on the screen, showing bright fringes (constructive interference) and dark fringes (destructive interference).

The simulation uses the following equations:

- Path difference: Δx = d sinθ ≈ d y/L  (where d is the slit separation, y is the position on the screen, and L is the distance to the screen)
- Constructive interference: Δx = mλ (where m is an integer representing the order of the fringe, and λ is the wavelength)
- Destructive interference: Δx = (m + 1/2)λ

The animation shows the build-up of the interference pattern as more and more photons are detected.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
L = 1  # Distance to the screen (m)
d = 0.001  # Slit separation (m)
wavelength = 650e-9  # Wavelength of light (m)
num_photons = 10000  # Number of photons to simulate
screen_width = 0.01  # Width of the screen (m)

# Create screen grid
x = np.linspace(-screen_width / 2, screen_width / 2, 500)

# Initialize intensity array
intensity = np.zeros_like(x)

# Animation function
def animate(i):
    # Simulate a single photon detection
    y = np.random.uniform(-screen_width / 2, screen_width / 2)  # Random position on the screen
    theta = np.arcsin(y / L)  # Angle of the photon
    delta_x = d * np.sin(theta)  # Path difference
    phase_diff = 2 * np.pi * delta_x / wavelength  # Phase difference
    intensity_contribution = np.cos(phase_diff)**2  # Intensity contribution from this photon

    # Update intensity array
    index = np.argmin(np.abs(x - y))
    intensity[index] += intensity_contribution

    # Plot the intensity
    line.set_data(x, intensity)
    return line,

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x, intensity)
ax.set_xlabel('Position (m)')
ax.set_ylabel('Intensity (arb. units)')
ax.set_title("Young's Double Slit Experiment")

# Create the animation
ani = FuncAnimation(fig, animate, frames=num_photons, interval=1, blit=True)

# Show the animation
if __name__ == "__main__":
    plt.show()
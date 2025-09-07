import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Constants ---
k = 10.0  # Spring constant in N/m
m = 1.0   # Mass in kg
A = 0.5   # Amplitude in meters

# --- 2. Calculate Parameters ---
# Angular frequency (omega) = sqrt(k/m)
omega = np.sqrt(k / m)
# Period (T) = 2 * pi / omega
period = 2 * np.pi / omega

# --- 3. Generate Time Data ---
# Create a time array from 0 to 5 seconds with 500 points for a smooth curve
t = np.linspace(0, 5, 500)

# --- 4. Calculate Displacement ---
# Displacement x at time t, assuming phase constant is 0
x = A * np.cos(omega * t)

# --- 5. Plot the Results ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, x, label=f'k={k}, m={m}, A={A}')

# Add titles and labels for clarity
ax.set_title('Simple Harmonic Motion of a Mass-Spring System', fontsize=16)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Displacement (m)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Set plot limits to focus on the motion
ax.set_xlim(0, 5)
ax.set_ylim(-A - 0.1, A + 0.1)

# --- 6. Output ---
print(f"The period of oscillation is: {period:.4f} seconds")
plt.show()

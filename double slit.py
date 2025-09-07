import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Educational Note:
# This simulation visualizes Young's double-slit experiment, a fundamental concept
# in wave optics. It demonstrates the principle of interference and the crucial
# role of coherence.
#
# 1. Coherent Light (e.g., from a laser):
#    - The waves from the two slits have a constant phase relationship.
#    - This creates a stable pattern of bright and dark fringes on the screen.
#    - Bright fringes (constructive interference) occur where waves add up in phase.
#    - Dark fringes (destructive interference) occur where waves add up out of phase.
#
# 2. Incoherent Light (e.g., from a light bulb):
#    - The phase relationship between the waves from the two slits changes randomly
#      and rapidly.
#    - At any single instant, an interference pattern exists, but it shifts position
#      randomly from one moment to the next.
#    - Our eyes or a detector average these rapidly shifting patterns over time.
#    - The result is that the fringes are "washed out," and we see only a smooth
#      distribution of light, which is the sum of the intensities from each
#      individual slit.

# --- Simulation Parameters ---
wavelength = 500e-9  # Wavelength of light (500 nm - green)
slit_separation = 0.1e-3  # Distance between the two slits (0.1 mm)
screen_distance = 2.0  # Distance from the slits to the screen (2 m)
screen_width = 10e-2  # Width of the observation screen (10 cm)
num_points = 1000  # Number of points on the screen for calculation
total_frames = 400  # Total frames in the animation
switch_frame = 200  # Frame number to switch from coherent to incoherent

# --- Setup the Figure and Axes ---
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

# Main plot for the intensity pattern
ax1 = fig.add_subplot(gs[0])
# Schematic plot of the experiment setup
ax2 = fig.add_subplot(gs[1])

# Create the screen coordinates
y_screen = np.linspace(-screen_width / 2, screen_width / 2, num_points)

# Pre-calculate path lengths
# Using the small angle approximation is common, but here we calculate the exact
# path lengths for better accuracy.
r1 = np.sqrt(screen_distance**2 + (y_screen - slit_separation / 2)**2)
r2 = np.sqrt(screen_distance**2 + (y_screen + slit_separation / 2)**2)
k = 2 * np.pi / wavelength  # Wavenumber

# --- Initialize Plot Elements ---
# Line for the instantaneous intensity pattern
line_inst, = ax1.plot(y_screen * 100, np.zeros_like(y_screen), lw=1, color='deepskyblue', label='Instantaneous Intensity')
# Line for the time-averaged intensity pattern
line_avg, = ax1.plot(y_screen * 100, np.zeros_like(y_screen), lw=3, color='royalblue', label='Time-Averaged Intensity')

ax1.set_title("Coherent Light", fontsize=16)
ax1.set_xlabel("Position on Screen (cm)", fontsize=12)
ax1.set_ylabel("Relative Intensity", fontsize=12)
ax1.set_ylim(0, 4.5)
ax1.set_xlim(y_screen.min() * 100, y_screen.max() * 100)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- Draw the Schematic ---
ax2.set_title("Experiment Setup")
ax2.plot([-0.5, -0.5], [-slit_separation * 2000, slit_separation * 2000], 'k', lw=4) # Barrier
ax2.plot([-0.5, -0.5], [slit_separation/2 * 2000 + 5, slit_separation * 2000], 'k', lw=4)
ax2.plot([-0.5, -0.5], [-slit_separation/2 * 2000 - 5, -slit_separation * 2000], 'k', lw=4)
ax2.text(-1, 0, "Source", ha='center')
ax2.text(-0.5, slit_separation/2 * 2000, "Slit 1", ha='right', va='center')
ax2.text(-0.5, -slit_separation/2 * 2000, "Slit 2", ha='right', va='center')
ax2.plot([0, screen_distance], [0, 0], 'k--') # Center line
screen_representation_x = screen_distance * np.ones_like(y_screen)
ax2.plot(screen_representation_x, y_screen * 50, 'grey', lw=8) # Screen
ax2.text(screen_distance, 0, "Screen", ha='left', va='center')
ax2.axis('off')
ax2.set_xlim(-1.5, screen_distance + 0.5)

fig.tight_layout()

# --- State variables for the animation ---
accumulated_intensity = np.zeros_like(y_screen)
frame_count_incoherent = 0

# --- Animation Functions ---
def init():
    """Initializes the animation plot elements."""
    line_inst.set_ydata(np.zeros_like(y_screen))
    line_avg.set_ydata(np.zeros_like(y_screen))
    return line_inst, line_avg

def update(frame):
    """
    This function is called for each frame of the animation.
    It calculates and updates the intensity patterns for coherent and
    incoherent light.
    """
    global accumulated_intensity, frame_count_incoherent

    # --- Part 1: Coherent Light Simulation ---
    if frame < switch_frame:
        # For coherent light, the phase difference is constant.
        # It only depends on the path difference from the slits to the screen.
        phase_difference = k * (r2 - r1)
        
        # Intensity I = I_0 * |E1 + E2|^2 = 4 * I_0 * cos^2(phase_difference / 2)
        # We assume a normalized base intensity (I_0 = 1).
        intensity = 4 * np.cos(phase_difference / 2)**2
        
        # In the coherent case, instantaneous and average intensities are the same.
        line_inst.set_ydata(intensity)
        line_avg.set_ydata(intensity)
        ax1.set_title(f"Coherent Light (Stable Interference Pattern)", fontsize=16)

    # --- Part 2: Incoherent Light Simulation ---
    else:
        if frame == switch_frame:
            # Reset accumulator when switching to incoherent mode
            accumulated_intensity = np.zeros_like(y_screen)
            frame_count_incoherent = 0

        frame_count_incoherent += 1
        
        # For incoherent light, the phase relationship is random.
        # We simulate this by adding a random phase shift in each "time step" (frame).
        random_phase = np.random.uniform(0, 2 * np.pi)
        phase_difference_inst = k * (r2 - r1) + random_phase
        
        # Calculate the instantaneous intensity. This is a shifting interference pattern.
        intensity_inst = 4 * np.cos(phase_difference_inst / 2)**2
        line_inst.set_ydata(intensity_inst)
        
        # Update the running average of the intensity. This simulates what a detector
        # or our eyes would perceive over time.
        accumulated_intensity += intensity_inst
        intensity_avg = accumulated_intensity / frame_count_incoherent
        line_avg.set_ydata(intensity_avg)
        
        ax1.set_title(f"Incoherent Light (Averaging over {frame_count_incoherent} patterns)", fontsize=16)

    return line_inst, line_avg

# --- Create and run the animation ---
ani = animation.FuncAnimation(fig, update, frames=total_frames,
                              init_func=init, blit=True, interval=50)

# To display the animation
plt.show()

import pygame
import math
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

# ========== User Configuration ==========

AUDIO_FILE = "preview_song.wav"  # Audio file to play (wav, flac, etc.)
SAMPLE_RATE = 44100              # Output audio sample rate (default: 44100)
CHANNELS = 2                     # Stereo output

# Assumed speed of sound (m/s) - not directly mapped to pixels, used for scaling
SPEED_OF_SOUND = 10.0

# Scaling factor for volume attenuation based on distance (in pixels)
DISTANCE_SCALE = 200.0

# Window size
WIDTH, HEIGHT = 800, 800
CENTER_X, CENTER_Y = WIDTH//2, HEIGHT//2  # Position of sound source

# ========== Load Audio File ==========
# Load the entire audio as a numpy array using soundfile
# Supports both mono and stereo files. Assumes mono for simplicity in this example.
data, sr_file = sf.read(AUDIO_FILE, dtype='float32')
if data.ndim == 1:
    # If mono, expand to (samples, 1) shape
    data = np.expand_dims(data, axis=1)
num_samples = data.shape[0]
num_channels_in_file = data.shape[1]

print(f"Loaded audio file: {AUDIO_FILE}")
print(f" - Sample rate: {sr_file}")
print(f" - Samples: {num_samples}")
print(f" - Channels: {num_channels_in_file}")

# Resampling is needed if sample rates differ, but this example assumes they match

# ========== Global Parameters (Updated in Real-Time) ==========
# Note: To be thread-safe between callback and main loop, proper locking should be used.
# For simplicity, this example uses global parameters and read-only access in the callback.
global_params = {
    "speed_factor": 1.0,  # Playback speed based on Doppler effect (affects pitch)
    "volume": 1.0,        # Volume after distance attenuation
    "pan": 0.0,           # Stereo pan: 1 (left) ~ 0 (right)
    "playhead": 0.0       # Current playback position (float, for interpolation)
}

# ========== Audio Callback Function ==========

def audio_callback(outdata, frames, time_info, status):
    """
    Streaming callback for sounddevice.
    outdata: numpy array of shape (frames, CHANNELS), dtype float32
    frames: number of frames to output this callback
    """
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)

    # Get global parameters
    spd = global_params["speed_factor"]
    vol = global_params["volume"]
    pan = global_params["pan"]

    # Pan mapping:
    # pan = -1 => left=1, right=0
    # pan =  0 => left=0.7, right=0.7 (center)
    # pan = +1 => left=0, right=1
    left_gain = vol * (1 - pan)      # More pan to right => lower left
    right_gain = vol * (pan)         # More pan to left => lower right

    # Zero out the output buffer
    outdata[:] = 0.0

    # Get current playhead position
    pos = global_params["playhead"]

    for i in range(frames):
        # 1) Fetch audio sample corresponding to playhead
        #    Assumes mono (uses channel 0) and expands to stereo manually
        
        idx_int = int(math.floor(pos))               # Integer index
        idx_next = (idx_int + 1) % num_samples       # Next sample (for interpolation)
        frac = pos - idx_int                         # Fractional part

        # Linear interpolation between current and next sample
        sample_current = data[idx_int % num_samples, 0]
        sample_next = data[idx_next, 0]
        samp = (1 - frac) * sample_current + frac * sample_next

        # 2) Apply gains and assign to stereo channels
        outdata[i, 0] = samp * left_gain   # Left channel
        outdata[i, 1] = samp * right_gain  # Right channel

        # 3) Advance playhead based on speed factor
        pos += spd
        if pos >= num_samples:
            pos -= num_samples  # Looping

    # Update global playhead
    global_params["playhead"] = pos


# ========== Start Audio Stream ==========

# Use sounddevice.OutputStream with callback-based streaming
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=1024,  # Buffer size
    channels=CHANNELS,
    dtype='float32',
    callback=audio_callback
)
stream.start()


# ========== Setup Pygame Window ==========

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SoundDevice Doppler Demo")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Previous mouse distance (for velocity estimation)
prev_distance = 200


# ========== Main Loop ==========

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stream.stop()
            stream.close()
            pygame.quit()
            sys.exit()

    mouse_x, mouse_y = pygame.mouse.get_pos()

    # --- Compute distance, angle, and speed of observer (mouse) ---
    dx = mouse_x - CENTER_X
    dy = mouse_y - CENTER_Y
    distance = math.sqrt(dx*dx + dy*dy)

    # Estimate observer speed (pixels per frame)
    observer_speed = distance - prev_distance

    # --- Doppler effect (playback speed) ---
    # Using formula: c / (c - v_o), ignoring units (pixels not meters)
    doppler_factor = (SPEED_OF_SOUND - 0.05 * observer_speed) / SPEED_OF_SOUND
    doppler_factor = max(min(doppler_factor, 3.0), 0.5)  # Clamp range

    # --- Volume attenuation based on distance ---
    # Formula: 1 / (distance / DISTANCE_SCALE), capped to avoid infinity at 0
    volume = 1.0 / max(distance / DISTANCE_SCALE, 1.0)

    # --- Stereo panning based on angle ---
    angle = math.atan2(dy, dx)  # -π to +π
    # Map angle to pan value
    pan_value = abs(angle / math.pi)*0.9 + 0.05

    # Update global parameters
    global_params["speed_factor"] = doppler_factor
    global_params["volume"] = volume
    global_params["pan"] = pan_value

    # --- Visual display ---
    screen.fill((30, 30, 30))

    # Draw sound source (center)
    pygame.draw.circle(screen, (255, 0, 0), (CENTER_X, CENTER_Y), 10)
    # Draw observer (mouse)
    pygame.draw.circle(screen, (0, 255, 0), (mouse_x, mouse_y), 5)

    info1 = font.render(
        f"Dist={distance:.2f}  Vol={volume:.2f}  Doppler={doppler_factor:.2f}",
        True, (255, 255, 255)
    )
    info2 = font.render(f"Observer Speed={observer_speed:.2f}  Pan={pan_value:.2f}", True, (255, 255, 255))
    screen.blit(info1, (20, 20))
    screen.blit(info2, (20, 50))

    pygame.display.flip()
    clock.tick(60)

    # Update previous distance
    prev_distance = distance

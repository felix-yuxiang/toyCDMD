import imageio
import os

# Directory containing the images
image_dir = "../outputs"  # Change this to your directory path

# List of epochs in order
epochs = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]

# Build the list of image paths in order
image_paths = [
    os.path.join(image_dir, f"toyexample_generated_samples_swiss_epoch_{epoch}.png")
    for epoch in epochs
]

# Read images
images = [imageio.imread(path) for path in image_paths]

# Save as GIF (duration is in seconds per frame)
imageio.mimsave("toyexample_swiss_training.gif", images, duration=10.0, loop=0)

print("GIF saved as toyexample_swiss_training.gif")
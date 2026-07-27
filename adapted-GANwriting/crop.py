import os
from PIL import Image

# Input and output folders
input_folder = "img2"
output_folder = "img3"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(input_path) as img:
            # Crop: left, top, right, bottom
            cropped = img.crop((0, 192, 1024, 576))
            cropped.save(output_path)

print("Done! All images have been cropped and saved to 'img3'.")
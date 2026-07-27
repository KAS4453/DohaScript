from PIL import Image
import os

# Input and output folders
input_folder = "img4"
output_folder = "img5"

os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

TOP_CROP = 65
BOTTOM_CROP = 65

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(extensions):
        continue

    image_path = os.path.join(input_folder, filename)

    try:
        img = Image.open(image_path)
        width, height = img.size

        if height <= TOP_CROP + BOTTOM_CROP:
            print(f"Skipping {filename}: Image height ({height}) is too small.")
            continue

        # Crop: (left, upper, right, lower)
        cropped = img.crop((0, TOP_CROP, width, height - BOTTOM_CROP))

        output_path = os.path.join(output_folder, filename)
        cropped.save(output_path)

        print(f"Processed {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Done!")
import os
from PIL import Image, ImageOps

input_dir = "img5"
output_dir = "img6"

os.makedirs(output_dir, exist_ok=True)

extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

threshold = 128  # Adjust between 100-180 if needed

for filename in os.listdir(input_dir):
    if filename.lower().endswith(extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            img = Image.open(input_path).convert("L")

            # Invert
            img = ImageOps.invert(img)

            # Binarize
            img = img.point(lambda p: 255 if p > threshold else 0)

            img.save(output_path)

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Finished.")
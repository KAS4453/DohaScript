from PIL import Image
import os

# Input and output folders
input_folder = "img3"
output_folder = "img4"

os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Process every image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(extensions):
        continue

    image_path = os.path.join(input_folder, filename)

    try:
        img = Image.open(image_path)
        width, height = img.size

        # Ensure image is of expected size
        if width != 1024 or height != 384:
            print(f"Skipping {filename}: Expected 1024x384, got {width}x{height}")
            continue

        crop_width = width // 4      # 256
        crop_height = height // 2    # 192

        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        count = 1
        for row in range(2):
            for col in range(4):
                left = col * crop_width
                upper = row * crop_height
                right = left + crop_width
                lower = upper + crop_height

                crop = img.crop((left, upper, right, lower))

                output_name = f"{base_name}_{count}{ext}"
                crop.save(os.path.join(output_folder, output_name))

                count += 1

        print(f"Processed {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Done!")
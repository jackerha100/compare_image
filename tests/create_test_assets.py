from PIL import Image, ImageDraw, ImageFont
import os

# Ensure data directory exists
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Common font (use a system available one, or bundle one for true portability)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 24)
except IOError:
    try:
        FONT = ImageFont.truetype("arial.ttf", 24) # Windows fallback
    except IOError:
        FONT = ImageFont.load_default() # Basic fallback

def create_image(path: str, size: tuple[int, int], color: str, text: str | None = None, elements: list[dict] | None = None):
    """Helper to create and save an image."""
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    
    if text:
        # Calculate text position for centering
        try:
            if hasattr(draw, 'textbbox'): # Pillow 9.2.0+
                 bbox = draw.textbbox((0,0), text, font=FONT, anchor="lt")
                 text_width = bbox[2] - bbox[0]
                 text_height = bbox[3] - bbox[1]
            else: # Older Pillow
                 text_width, text_height = draw.textsize(text, font=FONT)
        except Exception: # Fallback if font issues
            text_width, text_height = (len(text) * 10, 20) # Rough estimate

        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        draw.text((x, y), text, fill="black", font=FONT)

    if elements:
        for el in elements:
            draw.rectangle(el['bbox'], fill=el['color'])
            
    img.save(path)
    print(f"Created: {path}")

def main():
    # 1. data/test_image_valid.png
    create_image(os.path.join(DATA_DIR, "test_image_valid.png"), (100, 100), "blue")

    # 2. data/test_image_valid.jpg
    create_image(os.path.join(DATA_DIR, "test_image_valid.jpg"), (100, 100), "green")

    # 3. data/test_image_large.png
    create_image(os.path.join(DATA_DIR, "test_image_large.png"), (800, 600), "red")

    # 4. data/test_image_text.png
    create_image(os.path.join(DATA_DIR, "test_image_text.png"), (300, 100), "lightgray", text="Hello World")

    # Base for ref, missing, moved, diff_color
    ref_size = (200, 200)
    element_bbox_orig = (50, 50, 100, 100) # x1, y1, x2, y2
    
    # 5. data/test_image_ref.png
    create_image(os.path.join(DATA_DIR, "test_image_ref.png"), ref_size, "white", 
                 elements=[{'bbox': element_bbox_orig, 'color': 'black'}])

    # 6. data/test_image_missing_element.png
    create_image(os.path.join(DATA_DIR, "test_image_missing_element.png"), ref_size, "white", elements=[]) # No element

    # 7. data/test_image_moved_element.png
    element_bbox_moved = (120, 70, 170, 120)
    create_image(os.path.join(DATA_DIR, "test_image_moved_element.png"), ref_size, "white",
                 elements=[{'bbox': element_bbox_moved, 'color': 'black'}])

    # 8. data/test_image_different_color.png
    create_image(os.path.join(DATA_DIR, "test_image_different_color.png"), ref_size, "white",
                 elements=[{'bbox': element_bbox_orig, 'color': 'purple'}])

    # 9. data/invalid_image.png (dummy text file)
    invalid_path = os.path.join(DATA_DIR, "invalid_image.png")
    with open(invalid_path, "w") as f:
        f.write("This is not a valid PNG file.")
    print(f"Created: {invalid_path}")
    
    # 10. Create tests/output directory
    TEST_OUTPUT_DIR = "tests/output"
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
        print(f"Created directory: {TEST_OUTPUT_DIR}")
        # Create .gitignore in tests/output
        with open(os.path.join(TEST_OUTPUT_DIR, ".gitignore"), "w") as f:
            f.write("*\n!.gitignore\n")
        print(f"Created: {os.path.join(TEST_OUTPUT_DIR, '.gitignore')}")


if __name__ == "__main__":
    main()

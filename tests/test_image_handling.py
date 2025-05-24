import unittest
from unittest.mock import patch
import os
from PIL import Image, ImageDraw, ImageFont

# Modules to be tested
from src import image_loader as il
from src import image_processor as ip

# Global paths
DATA_DIR = "data"
TEST_OUTPUT_DIR = "tests/output"

# Ensure data and test_output directories exist (though create_test_assets.py should have made them)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TEST_OUTPUT_DIR):
    os.makedirs(TEST_OUTPUT_DIR)

# Helper to create an image if it doesn't exist (used in setUpClass)
def _ensure_image_exists(path: str, size: tuple[int, int], color: str, text: str | None = None, elements: list[dict] | None = None):
    if os.path.exists(path):
        return
    
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    
    if text:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        try:
            if hasattr(draw, 'textbbox'): # Pillow 9.2.0+
                 bbox = draw.textbbox((0,0), text, font=font, anchor="lt")
                 text_width = bbox[2] - bbox[0]
                 text_height = bbox[3] - bbox[1]
            else: # Older Pillow
                 text_width, text_height = draw.textsize(text, font=font)
        except Exception: 
            text_width, text_height = (len(text) * 10, 20) 

        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        draw.text((x, y), text, fill="black", font=font)

    if elements:
        for el in elements:
            draw.rectangle(el['bbox'], fill=el['color'])
            
    img.save(path)
    # print(f"Test setup created: {path}") # Optional: for debugging setup

class TestImageLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create images if they don't exist (idempotent setup)
        _ensure_image_exists(os.path.join(DATA_DIR, "test_image_valid_loader.png"), (100, 100), "blue")
        _ensure_image_exists(os.path.join(DATA_DIR, "test_image_valid_loader.jpg"), (100, 100), "green")
        
        # Create invalid_image.png for test_load_image_invalid_format
        cls.invalid_image_path = os.path.join(DATA_DIR, "invalid_image_loader.png")
        with open(cls.invalid_image_path, "w") as f:
            f.write("This is not a valid PNG file.")

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy invalid file
        if os.path.exists(cls.invalid_image_path):
            os.remove(cls.invalid_image_path)

    def test_load_image_valid_png(self):
        img = il.load_image(os.path.join(DATA_DIR, "test_image_valid_loader.png"))
        self.assertIsNotNone(img)
        self.assertIsInstance(img, Image.Image)

    def test_load_image_valid_jpg(self):
        img = il.load_image(os.path.join(DATA_DIR, "test_image_valid_loader.jpg"))
        self.assertIsNotNone(img)
        self.assertIsInstance(img, Image.Image)

    @patch('src.image_loader.print') # Mock print used for error logging in image_loader
    def test_load_image_not_found(self, mock_print):
        img = il.load_image(os.path.join(DATA_DIR, "non_existent_image.png"))
        self.assertIsNone(img)
        mock_print.assert_called_with(f"Error: Image file not found at {os.path.join(DATA_DIR, 'non_existent_image.png')}")

    @patch('src.image_loader.print') # Mock print used for error logging
    def test_load_image_invalid_format(self, mock_print):
        img = il.load_image(self.invalid_image_path)
        self.assertIsNone(img)
        mock_print.assert_called_with(f"Error: Could not open or read image file at {self.invalid_image_path}")


class TestImageProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create images if they don't exist
        cls.large_image_path = os.path.join(DATA_DIR, "test_image_large_processor.png")
        cls.small_image_path = os.path.join(DATA_DIR, "test_image_small_processor.png") # Ensure it's color
        _ensure_image_exists(cls.large_image_path, (800, 600), "red")
        _ensure_image_exists(cls.small_image_path, (100, 100), "cyan")


    def setUp(self):
        # Load images for each test
        self.large_image = il.load_image(self.large_image_path)
        self.small_image = il.load_image(self.small_image_path)
        self.assertIsNotNone(self.large_image, "Setup failed to load large_image")
        self.assertIsNotNone(self.small_image, "Setup failed to load small_image")
        # Ensure small_image is RGB for grayscale test consistency
        if self.small_image.mode != 'RGB':
            self.small_image = self.small_image.convert('RGB')


    def test_resize_image(self):
        max_dimension = 200
        resized_image = ip.resize_image(self.large_image, max_dimension)
        self.assertIsInstance(resized_image, Image.Image)
        
        original_width, original_height = self.large_image.size # 800, 600
        new_width, new_height = resized_image.size
        
        # Aspect ratio: 800/600 = 4/3
        expected_height_if_width_is_200 = 200 * (original_height / original_width) # 200 * (600/800) = 200 * 0.75 = 150
        expected_width_if_height_is_200 = 200 * (original_width / original_height) # 200 * (800/600) = 200 * 1.333 = 266.66
        
        self.assertTrue(new_width == max_dimension or new_height == max_dimension)
        if new_width == max_dimension: # Width is 200
            self.assertAlmostEqual(new_height, expected_height_if_width_is_200, delta=1)
        else: # Height is 200
            self.assertAlmostEqual(new_width, expected_width_if_height_is_200, delta=1)


    def test_resize_image_smaller_than_max(self):
        max_dimension = 200
        resized_image = ip.resize_image(self.small_image, max_dimension) # 100x100 image
        self.assertIsInstance(resized_image, Image.Image)
        self.assertEqual(resized_image.size, (100, 100)) # Dimensions should remain unchanged

    def test_convert_to_grayscale(self):
        # Ensure the input image is color (e.g., RGB)
        self.assertEqual(self.small_image.mode, 'RGB', "Small image for grayscale test should be RGB")
        
        grayscaled_image = ip.convert_to_grayscale(self.small_image)
        self.assertIsInstance(grayscaled_image, Image.Image)
        self.assertEqual(grayscaled_image.mode, 'L')

if __name__ == '__main__':
    unittest.main()
```

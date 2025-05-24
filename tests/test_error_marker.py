import unittest
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Modules to be tested
from src import error_marker as em
from src import image_loader as il

# Global paths
DATA_DIR = "data"
TEST_OUTPUT_DIR = "tests/output"

# Ensure data and test_output directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TEST_OUTPUT_DIR):
    os.makedirs(TEST_OUTPUT_DIR)

# Helper to create an image if it doesn't exist
def _create_image_if_missing(path: str, size: tuple[int, int], color: str):
    if os.path.exists(path):
        return
    img = Image.new("RGB", size, color)
    img.save(path)
    # print(f"Error marker test setup created: {path}")

def _ensure_test_images_exist_error_marker():
    _create_image_if_missing(os.path.join(DATA_DIR, "test_image_valid_marker.png"), (200, 200), "lightgreen")

class TestErrorMarker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_test_images_exist_error_marker()
        cls.base_image_path = os.path.join(DATA_DIR, "test_image_valid_marker.png")
        cls.base_image = il.load_image(cls.base_image_path)
        assert cls.base_image is not None, f"Failed to load base image for error marker tests: {cls.base_image_path}"

    def test_mark_errors_on_image_single_error(self):
        errors = [{
            'type': 'visual_discrepancy', # Use a type defined in ERROR_COLOR_MAP
            'bbox': (10, 10, 50, 50), 
            'description': 'A test error'
        }]
        
        marked_img = em.mark_errors_on_image(self.base_image, errors)
        
        self.assertIsNotNone(marked_img)
        self.assertIsInstance(marked_img, Image.Image)
        
        # Check if image is different (simple check: compare data arrays)
        # Convert to RGB to ensure consistent comparison, as mark_errors_on_image converts to RGB
        base_rgb = self.base_image.convert("RGB")
        marked_rgb = marked_img.convert("RGB")
        
        self.assertFalse(np.array_equal(np.array(base_rgb), np.array(marked_rgb)), 
                         "Marked image should be different from the base image.")
        
        # Optional: Check pixel color within the bbox.
        # This is more involved as text drawing might overlay the exact corner.
        # A simpler check is that the bounding box area is not the original background color.
        error_color_rgb = Image.new("RGB", (1,1), em.ERROR_COLOR_MAP.get('visual_discrepancy', 'red')).getpixel((0,0))
        
        # Check a pixel within the bounding box (e.g., top-left corner of the box outline)
        # Note: This might fail if text/background for text overlaps this exact pixel.
        # A more robust check would be to sample multiple points or average color.
        try:
            marked_pixel_color = marked_rgb.getpixel((10, 10)) # Top-left of bbox
            # This is a very strict check. Outline width, text, etc., can affect this.
            # self.assertEqual(marked_pixel_color, error_color_rgb, "Pixel color at bbox edge not as expected.")
        except Exception as e:
            self.fail(f"Could not get pixel color or comparison failed: {e}")

        if marked_img:
            marked_img.save(os.path.join(TEST_OUTPUT_DIR, "marked_single_error.png"))

    def test_mark_errors_on_image_multiple_errors(self):
        errors = [
            {'type': 'text_issue', 'bbox': (20, 30, 70, 60), 'description': 'Text error here'},
            {'type': 'layout_shift', 'bbox': (100, 110, 150, 160), 'description': 'Layout shifted'},
            {'type': 'default', 'bbox': (80, 10, 120, 40), 'description': 'Unknown issue'}
        ]
        marked_img = em.mark_errors_on_image(self.base_image, errors)
        self.assertIsNotNone(marked_img)
        self.assertIsInstance(marked_img, Image.Image)
        
        base_rgb = self.base_image.convert("RGB")
        marked_rgb = marked_img.convert("RGB")
        self.assertFalse(np.array_equal(np.array(base_rgb), np.array(marked_rgb)),
                         "Marked image with multiple errors should be different.")

        if marked_img:
            marked_img.save(os.path.join(TEST_OUTPUT_DIR, "marked_multiple_errors.png"))

    def test_mark_errors_on_image_no_errors(self):
        marked_img = em.mark_errors_on_image(self.base_image, [])
        self.assertIsNotNone(marked_img)
        
        # mark_errors_on_image returns an RGB copy if no errors.
        base_rgb_array = np.array(self.base_image.convert("RGB"))
        marked_rgb_array = np.array(marked_img.convert("RGB")) # marked_img is already RGB here
        
        self.assertTrue(np.array_equal(base_rgb_array, marked_rgb_array),
                        "Image with no errors marked should be identical to an RGB version of the original.")
        
        if marked_img:
            marked_img.save(os.path.join(TEST_OUTPUT_DIR, "marked_no_errors.png"))

    def test_mark_errors_invalid_bbox(self):
        errors = [{'type': 'test_error', 'bbox': (50, 50, 10, 10), 'description': 'Invalid bbox'}] # x2 < x1
        marked_img = em.mark_errors_on_image(self.base_image, errors)
        self.assertIsNotNone(marked_img)
        # Image should be same as original (or at least, no markings for this error)
        # mark_errors_on_image logs a warning and skips the error.
        base_rgb_array = np.array(self.base_image.convert("RGB"))
        marked_rgb_array = np.array(marked_img.convert("RGB"))
        self.assertTrue(np.array_equal(base_rgb_array, marked_rgb_array),
                        "Image with only invalid bbox error should be same as original.")

    def test_mark_errors_font_not_found_graceful_handling(self):
        original_font = em.FONT
        try:
            em.FONT = None # Simulate font not being found
            errors = [{'type': 'visual_discrepancy', 'bbox': (10, 10, 50, 50), 'description': 'A test error'}]
            marked_img = em.mark_errors_on_image(self.base_image, errors)
            self.assertIsNotNone(marked_img)
            # Box should still be drawn, even if text is not.
            # So image should differ from original.
            base_rgb = self.base_image.convert("RGB")
            marked_rgb = marked_img.convert("RGB")
            self.assertFalse(np.array_equal(np.array(base_rgb), np.array(marked_rgb)),
                             "Marked image (box only) should differ from base even if font is None.")
            if marked_img:
                marked_img.save(os.path.join(TEST_OUTPUT_DIR, "marked_font_not_found.png"))
        finally:
            em.FONT = original_font # Restore original font

if __name__ == '__main__':
    unittest.main()
```

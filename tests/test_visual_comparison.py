import unittest
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity # Used for direct comparison if needed

# Modules to be tested
from src import visual_differ as vd
from src import image_loader as il

# Global paths
DATA_DIR = "data"
TEST_OUTPUT_DIR = "tests/output" # For saving debug/visual output from tests

# Ensure data and test_output directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TEST_OUTPUT_DIR):
    os.makedirs(TEST_OUTPUT_DIR)

# Helper to create an image if it doesn't exist (used in setUpClass)
def _create_image_if_missing(path: str, size: tuple[int, int], color: str, text: str | None = None, elements: list[dict] | None = None):
    if os.path.exists(path):
        return
    
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    
    if text:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        try: # Basic text positioning
            if hasattr(draw, 'textbbox'): 
                 bbox = draw.textbbox((0,0), text, font=font, anchor="lt")
                 text_width = bbox[2] - bbox[0]
                 text_height = bbox[3] - bbox[1]
            else: 
                 text_width, text_height = draw.textsize(text, font=font)
        except Exception: 
            text_width, text_height = (len(text) * 10, 20) 

        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        draw.text((x, y), text, fill="black", font=font)

    if elements:
        for el in elements:
            draw.rectangle(el['bbox'], fill=el['color']) # el['bbox'] is (x1,y1,x2,y2)
            
    img.save(path)
    # print(f"Visual comparison test setup created: {path}")

def _ensure_test_images_exist_visual_comparison():
    ref_size = (200, 200)
    element_bbox_orig = (50, 50, 100, 100) 
    
    _create_image_if_missing(os.path.join(DATA_DIR, "test_image_ref.png"), ref_size, "white", 
                 elements=[{'bbox': element_bbox_orig, 'color': 'black'}])
    _create_image_if_missing(os.path.join(DATA_DIR, "test_image_missing_element.png"), ref_size, "white", elements=[])
    
    element_bbox_moved = (120, 70, 170, 120)
    _create_image_if_missing(os.path.join(DATA_DIR, "test_image_moved_element.png"), ref_size, "white",
                 elements=[{'bbox': element_bbox_moved, 'color': 'black'}])
    _create_image_if_missing(os.path.join(DATA_DIR, "test_image_different_color.png"), ref_size, "white",
                 elements=[{'bbox': element_bbox_orig, 'color': 'purple'}])


class TestVisualDiffer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_test_images_exist_visual_comparison()
        
        cls.img_ref = il.load_image(os.path.join(DATA_DIR, "test_image_ref.png"))
        cls.img_missing = il.load_image(os.path.join(DATA_DIR, "test_image_missing_element.png"))
        cls.img_moved = il.load_image(os.path.join(DATA_DIR, "test_image_moved_element.png"))
        cls.img_diff_color = il.load_image(os.path.join(DATA_DIR, "test_image_different_color.png"))
        cls.img_ref_copy = il.load_image(os.path.join(DATA_DIR, "test_image_ref.png")) # Fresh copy

        assert cls.img_ref is not None, "Failed to load test_image_ref.png"
        assert cls.img_missing is not None, "Failed to load test_image_missing_element.png"
        assert cls.img_moved is not None, "Failed to load test_image_moved_element.png"
        assert cls.img_diff_color is not None, "Failed to load test_image_different_color.png"
        assert cls.img_ref_copy is not None, "Failed to load test_image_ref.png (copy)"

    def test_compare_ssim_identical(self):
        score, diff_img_cv = vd.compare_ssim(self.img_ref, self.img_ref_copy)
        self.assertIsNotNone(score, "SSIM score should not be None for identical images.")
        self.assertAlmostEqual(score, 1.0, delta=0.01, msg="SSIM score for identical images should be very close to 1.0")
        self.assertIsNotNone(diff_img_cv, "SSIM difference image should not be None.")
        # For identical images, the diff image should be black (or very close to it).
        # A completely black image would have mean intensity of 0.
        # Due to minor float precision, it might be slightly above 0.
        self.assertTrue(np.mean(diff_img_cv) < 1.0, "Mean of SSIM diff image for identical images should be near 0.")
        if diff_img_cv is not None:
            Image.fromarray(diff_img_cv).save(os.path.join(TEST_OUTPUT_DIR, "debug_ssim_identical_diff.png"))

    def test_compare_ssim_different(self):
        score, diff_img_cv = vd.compare_ssim(self.img_ref, self.img_missing)
        self.assertIsNotNone(score, "SSIM score should not be None for different images.")
        self.assertLess(score, 0.95, "SSIM score for different images should be significantly less than 1.0") # Threshold is example
        self.assertIsNotNone(diff_img_cv, "SSIM difference image should not be None for different images.")
        self.assertTrue(np.mean(diff_img_cv) > 10, "Mean of SSIM diff image for different images should be significant.") # Expect some difference
        if diff_img_cv is not None:
            Image.fromarray(diff_img_cv).save(os.path.join(TEST_OUTPUT_DIR, "debug_ssim_different_diff.png"))

    def test_compare_orb_features_identical(self):
        matches, viz_img_cv = vd.compare_orb_features(self.img_ref, self.img_ref_copy, min_matches_threshold=5) # Lower threshold for small test img
        self.assertIsNotNone(matches, "ORB matches list should not be None.")
        # For simple test images, number of matches might be low. 
        # The exact number can be image dependent. Let's say > 10 for this specific test image.
        # The test_image_ref.png is simple, so ORB might find fewer than 20.
        self.assertTrue(len(matches) > 5, f"Expected more than 5 ORB matches for identical images, got {len(matches)}")
        self.assertIsNotNone(viz_img_cv, "ORB visualization image should not be None.")
        if viz_img_cv is not None:
            Image.fromarray(viz_img_cv[:, :, ::-1]).save(os.path.join(TEST_OUTPUT_DIR, "debug_orb_identical_matches.png"))


    def test_compare_orb_features_different(self):
        # Compare ref with missing element - should have very few matches
        matches_diff, viz_img_cv_diff = vd.compare_orb_features(self.img_ref, self.img_missing, min_matches_threshold=1)
        
        matches_identical, _ = vd.compare_orb_features(self.img_ref, self.img_ref_copy, min_matches_threshold=1)
        
        if matches_identical is not None and matches_diff is not None:
            self.assertLess(len(matches_diff), len(matches_identical) + 5,  # Allow some margin
                            "ORB matches for different images should be fewer than for identical ones.")
        elif matches_diff is not None: # identical failed for some reason, but diff still produced matches
            self.assertTrue(len(matches_diff) < 10, "Expected few ORB matches for different images.")
        else: # matches_diff is None
            pass # This implies 0 matches, which is expected for very different images or if one has no features.

        if viz_img_cv_diff is not None:
            Image.fromarray(viz_img_cv_diff[:, :, ::-1]).save(os.path.join(TEST_OUTPUT_DIR, "debug_orb_different_matches.png"))


    def test_compare_contours_identical(self):
        # compare_contours returns: contours_a, contours_b, diff_contours_on_b_img, diff_contours_list
        _, _, diff_img_cv, diff_contours = vd.compare_contours(self.img_ref, self.img_ref_copy)
        self.assertIsNotNone(diff_img_cv, "Contour difference image should not be None for identical images.")
        self.assertTrue(len(diff_contours) == 0 or all(cv2.contourArea(c) < 5 for c in diff_contours), 
                        "Difference contours for identical images should be empty or very small.")
        if diff_img_cv is not None:
            # The diff_img_cv from compare_contours has contours drawn on it.
            # To check if it's mostly black, we'd need the raw diff before drawing.
            # For now, we rely on diff_contours list.
            Image.fromarray(diff_img_cv[:, :, ::-1]).save(os.path.join(TEST_OUTPUT_DIR, "debug_contour_identical_diff.png"))

    def test_compare_contours_different(self):
        _, _, diff_img_cv, diff_contours = vd.compare_contours(self.img_ref, self.img_moved)
        self.assertIsNotNone(diff_img_cv, "Contour difference image should not be None for different images.")
        self.assertTrue(len(diff_contours) > 0, "Difference contours list should not be empty for different images.")
        # Check if there's at least one significant contour
        self.assertTrue(any(cv2.contourArea(c) > 20 for c in diff_contours), "Expected at least one significant difference contour.")
        if diff_img_cv is not None:
            Image.fromarray(diff_img_cv[:, :, ::-1]).save(os.path.join(TEST_OUTPUT_DIR, "debug_contour_different_diff.png"))

    def test_compare_average_colors_identical(self):
        similar, avg_a, avg_b = vd.compare_average_colors(self.img_ref, self.img_ref_copy)
        self.assertTrue(similar, "Average colors for identical images should be reported as similar.")
        self.assertIsNotNone(avg_a)
        self.assertIsNotNone(avg_b)

    def test_compare_average_colors_different(self):
        similar, avg_a, avg_b = vd.compare_average_colors(self.img_ref, self.img_diff_color)
        self.assertFalse(similar, "Average colors for images with different colored elements should be reported as different.")
        self.assertIsNotNone(avg_a)
        self.assertIsNotNone(avg_b)

if __name__ == '__main__':
    unittest.main()
```

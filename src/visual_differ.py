import cv2
import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _pil_to_cv2(pil_image: Image.Image, grayscale: bool = False) -> np.ndarray | None:
  """
  Converts a Pillow Image to an OpenCV image (NumPy array).

  Args:
    pil_image: The Pillow Image object.
    grayscale: If True, converts the image to grayscale.

  Returns:
    OpenCV image as a NumPy array, or None if input is None.
  """
  if pil_image is None:
    logging.error("Input Pillow image is None.")
    return None
  if grayscale:
    return np.array(pil_image.convert('L'))
  else:
    # Convert RGBA or P to RGB first if necessary
    if pil_image.mode in ('RGBA', 'P'):
        pil_image = pil_image.convert('RGB')
    return np.array(pil_image)[:, :, ::-1] # Convert RGB to BGR

def _cv2_to_pil(cv2_image: np.ndarray) -> Image.Image | None:
  """
  Converts an OpenCV image (NumPy array) to a Pillow Image.

  Args:
    cv2_image: The OpenCV image (NumPy array).

  Returns:
    Pillow Image object, or None if input is None.
  """
  if cv2_image is None:
    logging.error("Input OpenCV image is None.")
    return None
  if len(cv2_image.shape) == 2: # Grayscale
    return Image.fromarray(cv2_image, mode='L')
  else: # Color
    return Image.fromarray(cv2_image[:, :, ::-1], mode='RGB') # Convert BGR to RGB

def compare_ssim(image_a_pil: Image.Image, image_b_pil: Image.Image) -> tuple[float, np.ndarray | None]:
  """
  Compares two Pillow images using the Structural Similarity Index (SSIM).

  Args:
    image_a_pil: The first Pillow Image object.
    image_b_pil: The second Pillow Image object.

  Returns:
    A tuple containing the SSIM score (float) and 
    the difference image (NumPy array, scaled to 0-255 uint8), or (0.0, None) if comparison fails.
  """
  if image_a_pil is None or image_b_pil is None:
    logging.error("SSIM: One or both input images are None.")
    return 0.0, None

  image_a_gray_cv = _pil_to_cv2(image_a_pil.convert('L'), grayscale=True) # Ensure grayscale for SSIM
  image_b_gray_cv = _pil_to_cv2(image_b_pil.convert('L'), grayscale=True) # Ensure grayscale for SSIM
  
  if image_a_gray_cv is None or image_b_gray_cv is None:
    logging.error("SSIM: Failed to convert one or both images to OpenCV format.")
    return 0.0, None

  if image_a_gray_cv.shape != image_b_gray_cv.shape:
    logging.warning(
        f"SSIM: Image dimensions differ. Image A: {image_a_gray_cv.shape}, Image B: {image_b_gray_cv.shape}. "
        "Resizing Image B to match Image A for SSIM comparison."
    )
    image_b_gray_cv = cv2.resize(image_b_gray_cv, (image_a_gray_cv.shape[1], image_a_gray_cv.shape[0]), 
                               interpolation=cv2.INTER_AREA)

  try:
    # data_range is important if images are not in [0,1] float or [0,255] uint8.
    # scikit-image expects float images in range [-1, 1] or [0, 1] by default for SSIM.
    # If images are uint8, it handles data_range correctly.
    # Our _pil_to_cv2 converts to uint8.
    score, diff_image = structural_similarity(image_a_gray_cv, image_b_gray_cv, full=True, data_range=image_a_gray_cv.max() - image_a_gray_cv.min())
    # The difference image `diff_image` from SSIM is in the range [-1, 1] if inputs are float,
    # or can be in a different range if inputs are int. 
    # It needs to be rescaled to [0, 255] for visualization.
    diff_image = (diff_image - np.min(diff_image)) / (np.max(diff_image) - np.min(diff_image)) # Normalize to [0,1]
    diff_image = (diff_image * 255).astype("uint8")
    return float(score), diff_image
  except Exception as e:
    logging.error(f"SSIM: Error computing SSIM: {e}")
    return 0.0, None

def compare_orb_features(image_a_pil: Image.Image, image_b_pil: Image.Image, min_matches_threshold: int = 10) -> tuple[list[cv2.DMatch] | None, np.ndarray | None]:
  """
  Compares two Pillow images using ORB features.

  Args:
    image_a_pil: The first Pillow Image object.
    image_b_pil: The second Pillow Image object.
    min_matches_threshold: The minimum number of good matches to consider successful.
                           Note: The function also caps the number of visualized "good matches"
                           to a maximum of 50 for clarity.

  Returns:
    A tuple containing a list of good cv2.DMatch objects and an image (NumPy array) 
    visualizing the matches. Returns (None, None) if not enough matches are found or an error occurs.
  """
  if image_a_pil is None or image_b_pil is None:
    logging.error("ORB: One or both input images are None.")
    return None, None

  image_a_cv = _pil_to_cv2(image_a_pil, grayscale=True)
  image_b_cv = _pil_to_cv2(image_b_pil, grayscale=True)

  if image_a_cv is None or image_b_cv is None:
    logging.error("ORB: Failed to convert one or both images to OpenCV format.")
    return None, None

  try:
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image_a_cv, None)
    kp2, des2 = orb.detectAndCompute(image_b_cv, None)

    if des1 is None or des2 is None:
      logging.warning("ORB: No descriptors found for one or both images.")
      return None, None
    
    if len(des1) == 0 or len(des2) == 0:
      logging.warning("ORB: Zero descriptors found for one or both images.")
      return None, None


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep top N matches, e.g., 50, but only if they meet min_matches_threshold
    num_good_matches = min(len(matches), 50) 

    if len(matches) >= min_matches_threshold:
      good_matches = matches[:num_good_matches] # Take the top N best matches
      
      # For visualization, ensure images are color
      image_a_color_cv = _pil_to_cv2(image_a_pil, grayscale=False)
      image_b_color_cv = _pil_to_cv2(image_b_pil, grayscale=False)
      
      # Handle cases where conversion might fail or images are smaller than expected by drawMatches
      if image_a_color_cv is None or image_b_color_cv is None:
          logging.error("ORB: Failed to convert images to color for match visualization.")
          # Fallback: draw on grayscale if color conversion failed
          image_a_color_cv = image_a_cv
          image_b_color_cv = image_b_cv
          if image_a_color_cv is None or image_b_color_cv is None: # Still None, cannot draw
              return good_matches, None


      match_img = cv2.drawMatches(image_a_color_cv, kp1, image_b_color_cv, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
      return good_matches, match_img
    else:
      logging.info(f"ORB: Not enough good matches found ({len(matches)} < {min_matches_threshold}).")
      return None, None
  except cv2.error as e:
    logging.error(f"ORB: OpenCV error during feature matching: {e}")
    return None, None
  except Exception as e:
    logging.error(f"ORB: Unexpected error during feature matching: {e}")
    return None, None

def compare_contours(image_a_pil: Image.Image, image_b_pil: Image.Image, threshold_sensitivity: int = 128) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
  """
  Compares contours of two images by finding contours in their absolute difference.

  Args:
    image_a_pil: The first Pillow Image object.
    image_b_pil: The second Pillow Image object.
    threshold_sensitivity: Value used for simple thresholding if OTSU is not effective.
                           (Currently using OTSU, so this is a fallback idea).

  Returns:
    A tuple containing:
      - contours_a: List of contours from the first image's thresholded version.
      - contours_b: List of contours from the second image's thresholded version.
      - diff_contours_on_b_img: Image B with contours of differences drawn on it (NumPy array).
      - diff_contours: List of contours found in the difference image.
      Returns ([], [], None, []) if an error occurs.
  """
  if image_a_pil is None or image_b_pil is None:
    logging.error("Contours: One or both input images are None.")
    return [], [], None, []

  image_a_gray_cv = _pil_to_cv2(image_a_pil.convert('L'), grayscale=True)
  image_b_gray_cv = _pil_to_cv2(image_b_pil.convert('L'), grayscale=True)

  if image_a_gray_cv is None or image_b_gray_cv is None:
      logging.error("Contours: Failed to convert one or both images to grayscale OpenCV format.")
      return [], [], None, []
      
  # Ensure dimensions match for direct subtraction, resize B to A if necessary
  if image_a_gray_cv.shape != image_b_gray_cv.shape:
    logging.warning(
        f"Contours: Image dimensions differ. Image A: {image_a_gray_cv.shape}, Image B: {image_b_gray_cv.shape}. "
        "Resizing Image B to match Image A for contour difference."
    )
    image_b_gray_cv = cv2.resize(image_b_gray_cv, (image_a_gray_cv.shape[1], image_a_gray_cv.shape[0]), 
                               interpolation=cv2.INTER_AREA)


  # --- Find contours in individual images (optional, but part of spec) ---
  blur_a = cv2.GaussianBlur(image_a_gray_cv, (5, 5), 0)
  _, thresh_a = cv2.threshold(blur_a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  contours_a, _ = cv2.findContours(thresh_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  blur_b = cv2.GaussianBlur(image_b_gray_cv, (5, 5), 0)
  _, thresh_b = cv2.threshold(blur_b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  contours_b, _ = cv2.findContours(thresh_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # --- End of individual contour extraction ---

  # --- Find contours in the difference image ---
  abs_diff = cv2.absdiff(image_a_gray_cv, image_b_gray_cv)
  blur_diff = cv2.GaussianBlur(abs_diff, (5, 5), 0)
  
  # Use a fixed threshold or OTSU. OTSU might be problematic if differences are subtle.
  # Let's try a fixed threshold based on sensitivity, but ensure it's not too low.
  # A value like 30-50 might be a good starting point for `threshold_sensitivity`
  # if it were to be used as a fixed threshold directly.
  # For now, using OTSU on difference.
  _, thresh_diff = cv2.threshold(blur_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # If OTSU results in a threshold of 0 (common for identical images), use a fallback.
  # A better approach for `threshold_sensitivity` would be to use it if Otsu gives a very low/high value
  # or simply use a fixed threshold if Otsu is not desired.
  # For now, let's use a fixed threshold for differences if OTSU is too low.
  # if _ < 10: # If Otsu threshold is very low, it means images are very similar.
  #     _, thresh_diff = cv2.threshold(blur_diff, threshold_sensitivity, 255, cv2.THRESH_BINARY)


  diff_contours_list, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  diff_contours_list = diff_contours_list if diff_contours_list is not None else []


  # Draw difference contours on a color copy of image B
  image_b_color_cv = _pil_to_cv2(image_b_pil, grayscale=False)
  if image_b_color_cv is None:
      logging.error("Contours: Failed to convert image B to color for drawing difference contours.")
      # Fallback: try to draw on grayscale B if color conversion failed
      image_b_color_cv = image_b_gray_cv.copy() 
      # If it's grayscale, make it 3-channel so cv2.drawContours works with color
      if len(image_b_color_cv.shape) == 2:
          image_b_color_cv = cv2.cvtColor(image_b_color_cv, cv2.COLOR_GRAY2BGR)
      if image_b_color_cv is None: # Still None, cannot draw
          return contours_a or [], contours_b or [], None, []


  # Resize image_b_color_cv to match image_a_gray_cv's dimensions if they were different
  # This ensures that contours found on potentially resized image_b_gray_cv are drawn on an image of the same size
  if image_b_color_cv.shape[:2] != image_a_gray_cv.shape[:2]:
      image_b_color_cv = cv2.resize(image_b_color_cv, (image_a_gray_cv.shape[1], image_a_gray_cv.shape[0]),
                                    interpolation=cv2.INTER_AREA)


  cv2.drawContours(image_b_color_cv, diff_contours_list, -1, (0, 0, 255), 2) # Draw in red

  return contours_a or [], contours_b or [], image_b_color_cv, diff_contours_list


def compare_average_colors(image_a_pil: Image.Image, image_b_pil: Image.Image, tolerance: float = 0.1) -> tuple[bool, tuple[float, float, float] | None, tuple[float, float, float] | None]:
  """
  Compares the average RGB color of two Pillow images.

  Args:
    image_a_pil: The first Pillow Image object.
    image_b_pil: The second Pillow Image object.
    tolerance: The maximum allowed normalized difference for each RGB channel.
               (e.g., 0.1 means 10% difference).

  Returns:
    A tuple containing:
      - A boolean indicating if the average colors are similar within tolerance.
      - The average RGB tuple (R, G, B) for image A (0-255 range).
      - The average RGB tuple (R, G, B) for image B (0-255 range).
    Returns (False, None, None) if an error occurs.
  """
  if image_a_pil is None or image_b_pil is None:
    logging.error("AvgColor: One or both input images are None.")
    return False, None, None

  try:
    image_a_rgb = image_a_pil.convert('RGB')
    image_b_rgb = image_b_pil.convert('RGB')

    np_a = np.array(image_a_rgb)
    np_b = np.array(image_b_rgb)

    avg_color_a = tuple(np.mean(np_a, axis=(0, 1)))
    avg_color_b = tuple(np.mean(np_b, axis=(0, 1)))

    # Normalize differences
    diff_r = abs(avg_color_a[0] - avg_color_b[0]) / 255.0
    diff_g = abs(avg_color_a[1] - avg_color_b[1]) / 255.0
    diff_b = abs(avg_color_a[2] - avg_color_b[2]) / 255.0

    similar = (diff_r <= tolerance) and \
              (diff_g <= tolerance) and \
              (diff_b <= tolerance)

    return similar, avg_color_a, avg_color_b
  except Exception as e:
    logging.error(f"AvgColor: Error calculating average colors: {e}")
    return False, None, None

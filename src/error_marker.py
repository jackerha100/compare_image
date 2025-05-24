from PIL import Image, ImageDraw, ImageFont
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# Define a basic color map for error types
ERROR_COLOR_MAP = {
    "visual_discrepancy": "red",
    "text_issue": "blue",
    "missing_element": "yellow",
    "layout_shift": "green",
    "ocr_low_confidence": "cyan", 
    "gemini_issue": "purple",    
    "default": "magenta"
}

# Attempt to load a default font.
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 15)
except IOError:
    logging.warning(
        "Default font 'DejaVuSans.ttf' not found. "
        "Error descriptions will not be drawn on the image. "
        "Consider installing a common font or providing one with the application."
    )
    FONT = None


def mark_errors_on_image(image_pil: Image.Image, errors: list[dict]) -> Image.Image | None:
  """
  Marks detected errors on a copy of the input Pillow Image.

  Args:
    image_pil: The base Pillow Image object.
    errors: A list of dictionaries, where each dictionary represents an error.
            Expected keys in each error dict:
              - 'type' (str): The type of error (e.g., "visual_discrepancy").
                              Used to determine the color of the marking.
              - 'bbox' (tuple): A tuple (x1, y1, x2, y2) representing the
                                bounding box of the error.
              - 'description' (str, optional): Detailed information about the error.
                                               If provided and a font is available,
                                               this will be drawn near the bbox.
              - 'source' (str, optional): Information about what module detected the error.
                                          (e.g. "SSIM", "OCR", "Gemini")

  Returns:
    A new Pillow Image object with errors marked, or None if the input image is None.
    The returned image is always in RGB format.
  """
  if image_pil is None:
    logging.error("Input image is None. Cannot mark errors.")
    return None

  draw_image = image_pil.copy().convert("RGB")
  draw = ImageDraw.Draw(draw_image, 'RGBA') # Use RGBA for drawing context if using transparent backgrounds

  if not errors:
    logging.info("No errors provided to mark on the image.")
    return draw_image 

  logging.info(f"Marking {len(errors)} errors on the image.")

  for i, error in enumerate(errors):
    error_type = error.get('type', 'default')
    bbox = error.get('bbox')
    description = error.get('description')
    source = error.get('source', '') 

    if not bbox or not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
      logging.warning(f"Skipping error {i+1} due to missing or invalid 'bbox': {bbox}")
      continue

    try:
      x1, y1, x2, y2 = map(int, bbox) 
    except ValueError:
      logging.warning(f"Skipping error {i+1} due to non-integer values in 'bbox': {bbox}")
      continue
      
    if x1 >= x2 or y1 >= y2: 
        logging.warning(f"Skipping error {i+1} due to invalid bounding box dimensions (x1>=x2 or y1>=y2): {bbox}")
        continue

    color = ERROR_COLOR_MAP.get(error_type.lower(), ERROR_COLOR_MAP['default'])
    outline_width = 2 

    draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)

    text_to_draw = f"Type: {error_type}"
    if source:
        text_to_draw += f" ({source})"
    if description:
      text_to_draw += f"\nDesc: {description[:100]}" 

    if FONT and text_to_draw:
      text_x = x1
      text_y = y2 + 5  

      # Calculate text size for background and positioning
      try:
          if hasattr(draw, 'textbbox'): 
              # Get bounding box of the text to be drawn
              # Anchor 'la' means left, ascender (top for most fonts)
              text_box_for_size = draw.textbbox((text_x, text_y), text_to_draw, font=FONT, anchor='la')
              text_width = text_box_for_size[2] - text_box_for_size[0]
              text_height = text_box_for_size[3] - text_box_for_size[1]
          elif hasattr(FONT, 'getsize_multiline'): # Fallback for older Pillow
              text_size = FONT.getsize_multiline(text_to_draw)
              text_width = text_size[0]
              text_height = text_size[1]
          else: # Basic fallback (less accurate for multiline)
              text_size = draw.textsize(text_to_draw, font=FONT)
              text_width = text_size[0]
              text_height = text_size[1]
          
          # Adjust text_y if it flows off the bottom
          if text_y + text_height > draw_image.height:
              text_y = y1 - text_height - 5 
          
          # Adjust text_x and text_y if they are off-screen (e.g. negative)
          if text_y < 0: text_y = 5 
          if text_x < 0: text_x = 5
          if text_x + text_width > draw_image.width: # If text flows off the right
              text_x = draw_image.width - text_width - 5
              if text_x < 5: text_x = 5 # ensure it's not pushed too far left


          # Define background rectangle for text
          bg_padding = 3
          # The actual text_bbox for drawing will be based on the final text_x, text_y
          final_text_render_bbox = (text_x, text_y, text_x + text_width, text_y + text_height)

          bg_rect_x1 = final_text_render_bbox[0] - bg_padding
          bg_rect_y1 = final_text_render_bbox[1] - bg_padding
          bg_rect_x2 = final_text_render_bbox[2] + bg_padding
          bg_rect_y3 = final_text_render_bbox[3] + bg_padding
          
          # Create a temporary drawing surface for semi-transparent background
          # This ensures that the transparency of the background does not affect the main image's opacity
          # where it's not covered by this specific text background.
          text_bg_surface = Image.new('RGBA', draw_image.size, (0,0,0,0))
          temp_draw = ImageDraw.Draw(text_bg_surface)
          
          # Ensure background rectangle is within image bounds
          # This is less critical if text_bg_surface is used, as it's composited,
          # but good practice to avoid oversized intermediate surfaces if not needed.
          actual_bg_rect = [
              max(0, bg_rect_x1), max(0, bg_rect_y1),
              min(draw_image.width, bg_rect_x2), min(draw_image.height, bg_rect_y3)
          ]

          if actual_bg_rect[0] < actual_bg_rect[2] and actual_bg_rect[1] < actual_bg_rect[3]:
            temp_draw.rectangle(actual_bg_rect, fill=(255, 255, 255, 200)) # White with some transparency
          
          # Composite the text background onto the main draw_image
          # The mask ensures that only the non-transparent parts of text_bg_surface are pasted.
          draw_image.alpha_composite(text_bg_surface)
          # Re-initialize draw object on draw_image because alpha_composite may have changed its state
          # or if draw_image was not RGBA before, it would be now.
          # (draw_image was already converted to RGB, then draw context to RGBA, so this is fine)
          draw = ImageDraw.Draw(draw_image, 'RGBA')


          # Draw the text itself on top of its background
          # Using anchor 'la' (left, ascender) for more predictable positioning with textbbox.
          draw.text((text_x, text_y), text_to_draw, fill=color, font=FONT, anchor='la')

      except Exception as e:
          logging.warning(f"Could not draw text or text background for error {i+1}: {e}. Drawing box only.")
          # Fallback to drawing text without background if specific parts failed
          try:
            draw.text((text_x, text_y), text_to_draw, fill=color, font=FONT, anchor='la')
          except Exception as final_text_e:
            logging.error(f"Failed even fallback text drawing for error {i+1}: {final_text_e}")
          
  return draw_image

# Example usage
if __name__ == '__main__':
  try:
    test_image = Image.new("RGB", (600, 400), "lightgray")
    # Draw a diagonal line on the test image to see text positioning near edges
    draw_temp = ImageDraw.Draw(test_image)
    draw_temp.line([(0,0), (test_image.width, test_image.height)], fill="blue", width=5)
    draw_temp.line([(0,test_image.height), (test_image.width, 0)], fill="green", width=5)


    errors_to_mark = [
        {
            'type': 'visual_discrepancy', 'bbox': (20, 20, 120, 100), # Top-left
            'description': 'Color mismatch here. This is a long description to test text wrapping and background box.', 'source': 'SSIM'
        },
        {
            'type': 'text_issue', 'bbox': (480, 20, 580, 100), # Top-right
            'description': 'Text "Helo" should be "Hello".', 'source': 'OCR/Gemini'
        },
        {
            'type': 'missing_element', 'bbox': (20, 300, 120, 380), # Bottom-left
            'description': 'Icon missing.', 'source': 'Layout'
        },
        {
            'type': 'layout_shift', 'bbox': (480, 300, 580, 380), # Bottom-right
            'description': 'Button shifted.', 'source': 'VisualDiff'
        },
        { 
            'type': 'unknown_error_type', 'bbox': (250, 150, 350, 250), # Center
            'description': 'A centered error with a very very long description that will definitely need to be truncated or handled properly by the text drawing logic to see how it behaves.',
        },
        { 
            'type': 'visual_discrepancy', 'bbox': (50, 50, 40, 120), # Invalid bbox
            'description': 'Invalid box, should be skipped.'
        }
    ]
    marked_image = mark_errors_on_image(test_image, errors_to_mark)
    if marked_image:
      # Ensure the 'outputs' directory exists (create if not) for saving test images
      if not os.path.exists("outputs"):
          os.makedirs("outputs")
      output_path = "outputs/marked_test_image_error_marker.png"
      marked_image.save(output_path)
      logging.info(f"Saved example marked image to {output_path}")
    else:
      logging.error("Failed to create marked image in example.")

  except ImportError:
    logging.warning("Skipping __main__ example in error_marker.py due to missing 'os' module for output directory creation. This is unexpected.")
  except Exception as e:
    logging.error(f"Error in example usage of error_marker.py: {e}")

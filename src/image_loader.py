from PIL import Image

def load_image(image_path: str) -> Image.Image | None:
  """
  Loads an image from the given file path.
  Errors during loading (e.g., file not found, invalid format) are printed to stdout.

  Args:
    image_path: The path to the image file.

  Returns:
    A Pillow Image object if the image is loaded successfully, otherwise None.
  """
  try:
    img = Image.open(image_path)
    # It's good practice to load the image data after opening
    img.load() 
    return img
  except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    return None
  except IOError:
    print(f"Error: Could not open or read image file at {image_path}")
    return None

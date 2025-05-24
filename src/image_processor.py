from PIL import Image

def resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
  """
  Resizes an image while maintaining its aspect ratio, ensuring that neither
  width nor height exceeds max_dimension.
  If the image's dimensions are already smaller than or equal to max_dimension,
  the original image dimensions are maintained (though a new image object is created).

  Args:
    image: The Pillow Image object to resize.
    max_dimension: The maximum allowed dimension (width or height).

  Returns:
    The resized Pillow Image object.
  """
  original_width, original_height = image.size
  ratio = min(max_dimension / original_width, max_dimension / original_height)

  new_width = int(original_width * ratio)
  new_height = int(original_height * ratio)

  # Image.Resampling.LANCZOS is preferred for modern Pillow versions
  # Image.ANTIALIAS is for older versions (Pillow < 9.0.0)
  resampling_filter = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.ANTIALIAS
  
  resized_image = image.resize((new_width, new_height), resampling_filter)
  return resized_image

def convert_to_grayscale(image: Image.Image) -> Image.Image:
  """
  Converts a Pillow Image object to grayscale ('L' mode).

  Args:
    image: The Pillow Image object to convert.

  Returns:
    The grayscale Pillow Image object.
  """
  return image.convert('L')

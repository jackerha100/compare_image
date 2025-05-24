import pytesseract
from PIL import Image
import logging

# Assuming gemini_client.py is in the same directory or src.gemini_client can be resolved
try:
    from src import gemini_client
except ImportError:
    # Fallback for environments where src module is not directly recognized in this way
    import gemini_client 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# Reminder for Tesseract OCR installation
# For this code to work, Tesseract OCR must be installed on the system and
# pytesseract must be able to find it (e.g., tesseract executable in PATH).
# On Debian/Ubuntu: sudo apt-get install tesseract-ocr
# On macOS: brew install tesseract
# On Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki

MIN_CONFIDENCE_THRESHOLD = 30 # Filter out text blocks with confidence less than this

def extract_text_from_image(image_pil: Image.Image, lang: str = 'eng') -> list[dict]:
  """
  Extracts text from a Pillow Image object using Tesseract OCR.

  Args:
    image_pil: The Pillow Image object from which to extract text.
    lang: The language code for Tesseract (e.g., 'eng', 'fra').

  Returns:
    A list of dictionaries, where each dictionary represents a detected text
    block/word and contains: 'text', 'left', 'top', 'width', 'height', 'conf'.
    Returns an empty list if no text is found or an error occurs.
  """
  if image_pil is None:
    logging.error("Input image is None.")
    return []

  processed_text_data = []
  try:
    # image_to_data returns a dictionary with keys:
    # 'level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
    # 'left', 'top', 'width', 'height', 'conf', 'text'
    ocr_data = pytesseract.image_to_data(image_pil, lang=lang, output_type=pytesseract.Output.DICT)
    
    num_items = len(ocr_data['text'])

    for i in range(num_items):
      confidence = int(float(ocr_data['conf'][i])) # Confidence is often float string like '95.4321'
      text = ocr_data['text'][i].strip()

      if text and confidence >= MIN_CONFIDENCE_THRESHOLD:
        processed_text_data.append({
            'text': text,
            'left': ocr_data['left'][i],
            'top': ocr_data['top'][i],
            'width': ocr_data['width'][i],
            'height': ocr_data['height'][i],
            'conf': confidence
        })
    
    if not processed_text_data:
        logging.info(f"No text found with confidence >= {MIN_CONFIDENCE_THRESHOLD} in the image.")
    else:
        logging.info(f"Extracted {len(processed_text_data)} text segments from image.")

  except pytesseract.TesseractNotFoundError:
    logging.error(
        "Tesseract OCR is not installed or not found in your PATH. "
        "Please install Tesseract and ensure it's accessible."
    )
    # Optionally, re-raise the error if you want the main program to handle it explicitly
    # raise
    return []
  except Exception as e:
    logging.error(f"Error during Tesseract OCR processing: {e}")
    return []
  
  return processed_text_data


def analyze_text_coherence(
    text_data: list[dict], 
    # reference_image_pil: Image.Image | None = None, # Will be used later
    # test_image_pil: Image.Image | None = None # Will be used later
    ) -> list[dict]:
  """
  Analyzes the coherence of extracted text blocks, potentially using Gemini.

  Currently, this function iterates through `text_data` (assumed from a test image).
  If the Gemini client is configured and text is suitable (e.g., not empty, 
  confidence > 60), it calls `gemini_client.analyze_text_with_gemini()` for 
  each significant piece of text.

  Args:
    text_data: A list of text data dictionaries as returned by `extract_text_from_image`.
    # reference_image_pil: The reference Pillow Image object (currently unused).
    # test_image_pil: The test Pillow Image object (currently unused but text_data is from it).

  Returns:
    The augmented `text_data` list, with a 'gemini_analysis' key added to each
    dictionary for which analysis was performed.
  """
  if not text_data:
    logging.info("No text data provided for coherence analysis.")
    return []

  # Check if Gemini is configured. This relies on gemini_client.configure_gemini()
  # having been called successfully elsewhere (e.g., in main.py).
  # The gemini_client module itself should manage its configured state.
  # We can check a public variable or a status function if provided by gemini_client.
  # For now, we'll rely on analyze_text_with_gemini to handle unconfigured state.
  # A more robust way: if hasattr(gemini_client, 'is_configured') and not gemini_client.is_configured():
  #    logging.warning("Gemini client is not configured. Skipping Gemini analysis.")
  #    return text_data

  logging.info(f"Analyzing coherence for {len(text_data)} text segments.")
  
  for i, item in enumerate(text_data):
    text_to_analyze = item.get('text', '')
    confidence = item.get('conf', 0)
    
    # Analyze with Gemini if confidence is high and text is present
    if confidence > 60 and text_to_analyze:
      logging.info(f"Sending text to Gemini for analysis (confidence {confidence}): '{text_to_analyze[:100]}...'")
      # Ensure gemini_client is configured before calling
      # configure_gemini would ideally be called once at application startup.
      # If not, this is a fallback, but repeated calls are inefficient.
      # A better pattern is to ensure configuration at app start.
      # For this exercise, we assume it might not be, so try to configure.
      if not gemini_client._gemini_configured: # Accessing internal variable for check
          logging.warning("Gemini not pre-configured. Attempting to configure with environment variable.")
          gemini_client.configure_gemini() # Attempt to configure using env var

      if gemini_client._gemini_configured:
          analysis_result = gemini_client.analyze_text_with_gemini(text_to_analyze)
          if analysis_result:
            text_data[i]['gemini_analysis'] = analysis_result
          else:
            text_data[i]['gemini_analysis'] = "Gemini analysis failed or returned no content."
            logging.warning(f"Gemini analysis for text '{text_to_analyze[:50]}' returned None or failed.")
      else:
          text_data[i]['gemini_analysis'] = "Gemini client not configured. Analysis skipped."
          logging.warning(f"Skipping Gemini analysis for text '{text_to_analyze[:50]}' as client is not configured.")
          # If Gemini is essential, we might stop or handle differently.
          # For now, we just note it and continue.
    else:
      item['gemini_analysis'] = "Not analyzed (low confidence or no text)."

  return text_data

# Note: The optional `compare_text_elements` function is deferred as per instructions.
# Reminder: Gemini API key setup is crucial for `analyze_text_with_gemini` to work.
# The `gemini_client.configure_gemini()` function should be called (e.g., in main)
# with an API key or ensure the GOOGLE_API_KEY environment variable is set.

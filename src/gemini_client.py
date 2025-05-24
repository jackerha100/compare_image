import google.generativeai as genai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

_gemini_configured = False

def configure_gemini(api_key: str | None = None) -> bool:
  """
  Configures the Google Generative AI client (Gemini).

  Attempts to use the provided API key. If not provided, it tries to
  fetch the API key from the 'GOOGLE_API_KEY' environment variable.

  Args:
    api_key: The Google API key to use.

  Returns:
    True if configuration was successful, False otherwise.
  """
  global _gemini_configured
  if api_key:
    pass
  elif os.getenv('GOOGLE_API_KEY'):
    api_key = os.getenv('GOOGLE_API_KEY')
  else:
    logging.error("Gemini API key not provided and not found in GOOGLE_API_KEY environment variable.")
    _gemini_configured = False
    return False

  try:
    genai.configure(api_key=api_key)
    logging.info("Gemini client configured successfully.")
    _gemini_configured = True
    return True
  except Exception as e:
    logging.error(f"Failed to configure Gemini client: {e}")
    _gemini_configured = False
    return False

def analyze_text_with_gemini(text_to_analyze: str, model_name: str = "gemini-pro") -> str | None:
  """
  Analyzes a given text string using a Gemini generative model.

  Args:
    text_to_analyze: The text string to analyze.
    model_name: The name of the Gemini model to use (e.g., "gemini-pro").

  Returns:
    The text part of Gemini's response, or None if analysis fails or Gemini is not configured.
  """
  global _gemini_configured
  if not _gemini_configured:
    logging.error("Gemini client not configured. Please call configure_gemini() first.")
    return None

  if not text_to_analyze or text_to_analyze.strip() == "":
    logging.info("No text provided to analyze.")
    return "No text to analyze."

  try:
    model = genai.GenerativeModel(model_name)
    
    prompt = (
        "Please analyze the following text for potential issues. Consider if the text:\n"
        "- Is nonsensical, gibberish, or completely random characters.\n"
        "- Appears to be cropped, cut off, or abruptly terminated.\n"
        "- Seems to be the result of OCR errors on blurry or low-quality images (e.g., strange characters, mixed-up words).\n"
        "- Shows inconsistent use of fonts, sizes, or styles that make it hard to read (inferred from text patterns).\n"
        "- Lacks overall readability, coherence, or logical flow.\n\n"
        "Provide a brief summary of any identified issues. If the text seems generally okay, please state that. "
        "Focus on clear, concise observations. Here is the text:\n\n"
    )

    full_prompt = prompt + text_to_analyze
    
    # Gemini API can sometimes be sensitive to prompt length or content.
    # For very long text, consider chunking or summarizing first if issues arise.
    # However, for typical OCR outputs from image segments, it should be fine.

    response = model.generate_content(full_prompt)
    
    if response.parts:
      return response.text
    else:
      # Handle cases where the response might be blocked or have no parts
      logging.warning(f"Gemini response for text '{text_to_analyze[:50]}...' was empty or potentially blocked. Safety ratings: {response.prompt_feedback}")
      return "Analysis generated no specific feedback or response was blocked."

  except Exception as e:
    logging.error(f"Error during Gemini API call for text '{text_to_analyze[:50]}...': {e}")
    # Check for specific API errors if available in the exception object
    # For example, if hasattr(e, 'response') and e.response.status_code == 429 # Rate limit
    return f"Error analyzing text: {e}"

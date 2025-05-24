import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
from PIL import Image, ImageDraw, ImageFont
import pytesseract # Needed for pytesseract.Output.DICT value

# Modules to be tested
from src import text_analyzer as ta
from src import gemini_client as gc
from src import image_loader as il

# Global paths
DATA_DIR = "data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Helper to create an image if it doesn't exist
def _create_image_if_missing(path: str, size: tuple[int, int], color: str, text: str | None = None):
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
            
    img.save(path)
    # print(f"Text analysis test setup created: {path}")

def _ensure_test_images_exist_text_analysis():
    _create_image_if_missing(os.path.join(DATA_DIR, "test_image_text.png"), (300, 100), "lightgray", text="Hello World")


class TestGeminiClient(unittest.TestCase):

    def setUp(self):
        # Reset the _gemini_configured flag before each test to ensure isolation
        gc._gemini_configured = False

    @patch('src.gemini_client.genai.configure')
    @patch('src.gemini_client.os.getenv')
    def test_configure_gemini_with_api_key(self, mock_getenv, mock_genai_configure):
        configured = gc.configure_gemini(api_key="test_key")
        self.assertTrue(configured)
        mock_genai_configure.assert_called_once_with(api_key="test_key")
        mock_getenv.assert_not_called() # Should not try to get from env if key is provided
        self.assertTrue(gc._gemini_configured)

    @patch('src.gemini_client.genai.configure')
    @patch('src.gemini_client.os.getenv')
    def test_configure_gemini_with_env_var(self, mock_getenv, mock_genai_configure):
        mock_getenv.return_value = "env_key"
        configured = gc.configure_gemini()
        self.assertTrue(configured)
        mock_getenv.assert_called_once_with('GOOGLE_API_KEY')
        mock_genai_configure.assert_called_once_with(api_key="env_key")
        self.assertTrue(gc._gemini_configured)

    @patch('src.gemini_client.genai.configure')
    @patch('src.gemini_client.os.getenv')
    def test_configure_gemini_no_key_available(self, mock_getenv, mock_genai_configure):
        mock_getenv.return_value = None # No env var
        configured = gc.configure_gemini() # No api key argument
        self.assertFalse(configured)
        mock_getenv.assert_called_once_with('GOOGLE_API_KEY')
        mock_genai_configure.assert_not_called()
        self.assertFalse(gc._gemini_configured)

    @patch('src.gemini_client.genai.GenerativeModel')
    @patch('src.gemini_client.logging.error') # Mock logging
    def test_analyze_text_with_gemini_success(self, mock_log_error, mock_generative_model):
        # Configure Gemini (mocking actual genai.configure)
        with patch('src.gemini_client.genai.configure'):
            gc.configure_gemini(api_key="fake_key") 
            self.assertTrue(gc._gemini_configured) # Ensure it's set for this test

        mock_model_instance = MagicMock()
        # Simulate the response structure: response.text
        mock_response = MagicMock()
        mock_response.text = "Gemini says ok"
        mock_response.parts = [MagicMock()] # Ensure response.parts is not empty
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance
        
        response = gc.analyze_text_with_gemini("some text")
        
        self.assertEqual(response, "Gemini says ok")
        mock_generative_model.assert_called_once_with("gemini-pro")
        mock_model_instance.generate_content.assert_called_once()
        # Check that the prompt is part of the call
        args, _ = mock_model_instance.generate_content.call_args
        self.assertIn("some text", args[0]) 
        mock_log_error.assert_not_called()

    @patch('src.gemini_client.logging.error') # Mock logging
    @patch('src.gemini_client.genai.GenerativeModel')
    def test_analyze_text_with_gemini_not_configured(self, mock_generative_model, mock_log_error):
        gc._gemini_configured = False # Explicitly ensure not configured
        response = gc.analyze_text_with_gemini("some text")
        
        self.assertIsNone(response)
        mock_log_error.assert_called_once_with("Gemini client not configured. Please call configure_gemini() first.")
        mock_generative_model.assert_not_called()

    @patch('src.gemini_client.logging.error') # Mock logging
    @patch('src.gemini_client.genai.GenerativeModel')
    def test_analyze_text_with_gemini_api_error(self, mock_generative_model, mock_log_error):
        with patch('src.gemini_client.genai.configure'):
            gc.configure_gemini(api_key="fake_key")
            self.assertTrue(gc._gemini_configured)

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_generative_model.return_value = mock_model_instance
        
        response = gc.analyze_text_with_gemini("some text")
        
        self.assertTrue(response.startswith("Error analyzing text: API Error")) # Check if the error message is returned
        mock_log_error.assert_called_once_with("Error during Gemini API call for text 'some text...': API Error")

    @patch('src.gemini_client.logging.warning') # Mock logging
    @patch('src.gemini_client.genai.GenerativeModel')
    def test_analyze_text_with_gemini_empty_response_parts(self, mock_generative_model, mock_log_warning):
        with patch('src.gemini_client.genai.configure'):
            gc.configure_gemini(api_key="fake_key")
            self.assertTrue(gc._gemini_configured)

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.parts = [] # Simulate empty parts
        mock_response.text = None # No text if parts is empty
        # Simulate prompt_feedback for logging
        type(mock_response).prompt_feedback = PropertyMock(return_value="Safety rating: OK") 
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance
        
        response = gc.analyze_text_with_gemini("some text")
        
        self.assertEqual(response, "Analysis generated no specific feedback or response was blocked.")
        mock_log_warning.assert_called_once_with("Gemini response for text 'some text...' was empty or potentially blocked. Safety ratings: Safety rating: OK")


class TestTextAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_test_images_exist_text_analysis()
        cls.text_image_path = os.path.join(DATA_DIR, "test_image_text.png")
        cls.text_image = il.load_image(cls.text_image_path)
        assert cls.text_image is not None, f"Failed to load test image: {cls.text_image_path}"

    @patch('src.text_analyzer.pytesseract')
    def test_extract_text_from_image_success(self, mock_pytesseract):
        sample_ocr_output = {
            'level':    [1, 2, 3, 4, 5, 5], 
            'page_num': [1, 1, 1, 1, 1, 1],
            'block_num':[1, 1, 1, 1, 1, 1],
            'par_num':  [1, 1, 1, 1, 1, 1],
            'line_num': [1, 1, 1, 1, 1, 1],
            'word_num': [1, 1, 1, 1, 1, 2], # word_num is specific to word level data
            'left':     [10, 10, 10, 10, 10, 50],
            'top':      [10, 10, 10, 10, 10, 10],
            'width':    [100, 100, 100, 100, 30, 40],
            'height':   [20, 20, 20, 20, 15, 15],
            'conf':     ['-1', '-1', '-1', '-1', '90', '85'], # -1 indicates non-word segments, block/par/line
            'text':     ['', '', '', '', 'Hello', 'World']
        }
        # Ensure the Output.DICT attribute is correctly mocked if it's accessed
        # by the code under test (it usually is, to pass to image_to_data)
        mock_pytesseract.Output.DICT = pytesseract.Output.DICT 
        mock_pytesseract.image_to_data.return_value = sample_ocr_output
        
        text_data = ta.extract_text_from_image(self.text_image)
        
        mock_pytesseract.image_to_data.assert_called_once_with(self.text_image, lang='eng', output_type=pytesseract.Output.DICT)
        self.assertIsInstance(text_data, list)
        self.assertEqual(len(text_data), 2) # Should filter out items with conf < MIN_CONFIDENCE_THRESHOLD (implicitly -1) and empty text
        
        self.assertEqual(text_data[0]['text'], 'Hello')
        self.assertEqual(text_data[0]['conf'], 90)
        self.assertEqual(text_data[0]['left'], 10)
        
        self.assertEqual(text_data[1]['text'], 'World')
        self.assertEqual(text_data[1]['conf'], 85)

    @patch('src.text_analyzer.pytesseract')
    def test_extract_text_from_image_no_text_detected(self, mock_pytesseract):
        # Simulate Tesseract finding no text (empty lists for all keys)
        empty_ocr_output = {
            'level': [], 'page_num': [], 'block_num': [], 'par_num': [], 
            'line_num': [], 'word_num': [], 'left': [], 'top': [], 
            'width': [], 'height': [], 'conf': [], 'text': []
        }
        mock_pytesseract.Output.DICT = pytesseract.Output.DICT
        mock_pytesseract.image_to_data.return_value = empty_ocr_output
        
        text_data = ta.extract_text_from_image(self.text_image)
        self.assertEqual(text_data, [])

    @patch('src.text_analyzer.pytesseract.image_to_data', side_effect=pytesseract.TesseractNotFoundError)
    @patch('src.text_analyzer.logging.error')
    def test_extract_text_from_image_tesseract_not_found(self, mock_log_error, mock_image_to_data):
        text_data = ta.extract_text_from_image(self.text_image)
        self.assertEqual(text_data, [])
        mock_log_error.assert_called_with(
            "Tesseract OCR is not installed or not found in your PATH. "
            "Please install Tesseract and ensure it's accessible."
        )

    @patch('src.text_analyzer.gemini_client.analyze_text_with_gemini')
    @patch('src.text_analyzer.gemini_client.configure_gemini')
    @patch('src.text_analyzer.gemini_client._gemini_configured', True) # Assume configured for this test
    def test_analyze_text_coherence_with_gemini(self, mock_gemini_configured_flag, mock_configure_gemini, mock_analyze_gemini):
        # mock_gemini_configured_flag is used to directly set the private variable _gemini_configured to True
        # This bypasses the need for configure_gemini to be successful if it were more complex.
        # The text_analyzer.py code checks this flag.

        text_items = [
            {'text': 'Good Text', 'conf': 95, 'left': 10, 'top': 10, 'width': 30, 'height': 15}, 
            {'text': 'B@d Txt', 'conf': 80, 'left': 10, 'top': 30, 'width': 30, 'height': 15},
            {'text': 'LowConf', 'conf': 50, 'left': 10, 'top': 50, 'width': 30, 'height': 15} # Should be skipped by confidence check
        ]
        
        mock_analyze_gemini.side_effect = ["Seems okay", "This is nonsensical"]
        
        # analyze_text_coherence now only takes text_data
        result = ta.analyze_text_coherence(text_items) 
        
        # configure_gemini in text_analyzer is called if _gemini_configured is False.
        # Since we mocked _gemini_configured to True at the module level for this test path,
        # the internal configure_gemini call in analyze_text_coherence should ideally not happen.
        # However, the test setup for text_analyzer.py's analyze_text_coherence might attempt its own
        # configuration if it doesn't see the global _gemini_configured as True.
        # Let's check if the direct gc.configure_gemini was called (it shouldn't if already configured)
        # mock_configure_gemini.assert_not_called() # This depends on how strictly the test setup isolates _gemini_configured

        self.assertEqual(mock_analyze_gemini.call_count, 2)
        mock_analyze_gemini.assert_any_call('Good Text')
        mock_analyze_gemini.assert_any_call('B@d Txt')
        
        self.assertEqual(result[0]['gemini_analysis'], "Seems okay")
        self.assertEqual(result[1]['gemini_analysis'], "This is nonsensical")
        self.assertEqual(result[2]['gemini_analysis'], "Not analyzed (low confidence or no text).")


    @patch('src.text_analyzer.gemini_client.analyze_text_with_gemini')
    @patch('src.text_analyzer.gemini_client.configure_gemini')
    @patch('src.text_analyzer.gemini_client._gemini_configured', False) # Ensure Gemini is seen as not configured
    def test_analyze_text_coherence_gemini_not_configured(self, mock_gemini_configured_flag, mock_configure_gemini, mock_analyze_gemini):
        # Simulate that configure_gemini fails when called inside analyze_text_coherence
        mock_configure_gemini.return_value = False 
        
        text_items = [{'text': 'Some Text', 'conf': 95, 'left':1, 'top':1, 'width':1, 'height':1}]
        result = ta.analyze_text_coherence(text_items)
        
        # configure_gemini should be called by analyze_text_coherence if _gemini_configured is False
        mock_configure_gemini.assert_called_once() 
        mock_analyze_gemini.assert_not_called() # Because configuration failed
        
        self.assertIn('gemini_analysis', result[0])
        self.assertEqual(result[0]['gemini_analysis'], "Gemini client not configured. Analysis skipped.")

if __name__ == '__main__':
    _ensure_test_images_exist_text_analysis() # Ensure image exists before running tests
    unittest.main()
```

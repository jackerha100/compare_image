# Image Comparison Tool

This project develops a tool to compare a 'reference image' against a 'test image'. It automatically detects visual and structural differences, highlights these on the test image, and can leverage Google's Gemini API for advanced text analysis.

## Features

*   **Visual Discrepancy Detection:**
    *   Uses Structural Similarity Index (SSIM) to identify overall similarity and local differences.
    *   Compares ORB feature descriptors to detect structural changes.
    *   Analyzes image contours to find layout shifts and shape differences.
    *   Compares average image colors to detect global color changes.
*   **Text Analysis (OCR and AI):**
    *   Extracts text from images using Tesseract OCR.
    *   Analyzes extracted text for coherence, OCR errors, and formatting issues using Google Gemini (if API key is provided).
*   **Error Marking:**
    *   Draws bounding boxes and descriptions directly onto a copy of the test image to indicate detected errors.
*   **Technologies Used:**
    *   Python
    *   Pillow (PIL Fork) for image manipulation.
    *   OpenCV for advanced image processing tasks.
    *   Scikit-image for SSIM calculation.
    *   Pytesseract for Optical Character Recognition (OCR).
    *   Google Generative AI (Gemini) for AI-powered text analysis.

## Installation

**1. Python Version:**
   *   This project requires Python 3.8 or newer.

**2. Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

**3. Install Python Dependencies:**
   *   It's recommended to use a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   *   Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

**4. External Dependencies:**

   *   **Tesseract OCR Engine:**
        *   Tesseract OCR must be installed on your system and the `tesseract` executable must be in your system's PATH.
        *   **Installation Guides:** Visit [Tesseract OCR Documentation](https://github.com/tesseract-ocr/tessdoc) for instructions.
        *   **Language Data:** Ensure you have the English language data pack installed for Tesseract (usually `eng`). Other languages can be installed if needed for `pytesseract` by specifying the `lang` parameter in relevant functions (though the main script currently defaults to English).

   *   **Google Gemini API Key (Optional but Recommended for Text Analysis):**
        *   For advanced text analysis features, a Google Gemini API key is required.
        *   **Obtain an API Key:** You can get an API key from Google AI Studio: [https://aistudio.google.com/](https://aistudio.google.com/)
        *   **Set up the API Key:** You can provide the API key to the tool in one of two ways:
            1.  **Command-line Argument:** Use the `--gemini_api_key YOUR_API_KEY` argument when running the script.
            2.  **Environment Variable:** Set the `GOOGLE_API_KEY` environment variable to your API key.
                ```bash
                export GOOGLE_API_KEY="YOUR_API_KEY_HERE" # Linux/macOS
                # set GOOGLE_API_KEY="YOUR_API_KEY_HERE" # Windows Command Prompt
                # $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE" # Windows PowerShell
                ```
        *   If no API key is provided, text analysis will be limited to OCR extraction without Gemini's advanced coherence checks.

## Usage

The main script for the tool is `src/main.py`.

**Command-Line Arguments:**

*   `--ref_image <path>`: (Required) Path to the reference image file.
*   `--test_image <path>`: (Required) Path to the test image file.
*   `--output_image <path>`: (Required) Path to save the output image with marked differences.
*   `--gemini_api_key <key>`: (Optional) Your Google Gemini API key. If not provided, the tool will attempt to use the `GOOGLE_API_KEY` environment variable.

**Example Command:**

To compare a reference image with a test image and save the result:

```bash
python src/main.py \
  --ref_image data/sample_ref.png \
  --test_image data/sample_test_layout_issue.png \
  --output_image output/result_layout.png \
  --gemini_api_key YOUR_API_KEY_HERE
```

(Replace `YOUR_API_KEY_HERE` with your actual Gemini API key, or ensure the environment variable is set.)

## Output

*   **Marked-up Image:** The primary output is an image file (saved to the path specified by `--output_image`) which is a copy of the test image with detected differences and errors highlighted by bounding boxes and textual descriptions.
*   **Console Summary:** A summary of detected errors and observations is printed to the console after processing.
*   **Debug Images:** Additional debug images (e.g., SSIM difference map, ORB matches, contour differences) are saved in the same directory as the main output image. These are named with a `debug_` prefix (e.g., `debug_ssim_difference.png`).

## Project Structure

The project's source code is organized as follows:

*   `src/`: Contains the core logic of the application.
    *   `main.py`: The main command-line interface and orchestrator for the image comparison process.
    *   `image_loader.py`: Handles loading images from file paths.
    *   `image_processor.py`: Provides image preprocessing functions (resize, grayscale).
    *   `visual_differ.py`: Contains functions for various visual comparison techniques (SSIM, ORB, contours, color).
    *   `text_analyzer.py`: Handles OCR text extraction (using Pytesseract) and text coherence analysis (using Gemini).
    *   `gemini_client.py`: Manages interaction with the Google Gemini API.
    *   `error_marker.py`: Responsible for drawing error markings on images.
*   `data/`: Intended for placing input images. Includes sample images.
*   `output/`: Default directory for saving output images (will be created if it doesn't exist).
*   `tests/`: Contains unit tests for the project.
    *   `create_test_assets.py`: Script to generate various test images used by the unit tests.
    *   Individual `test_*.py` files for each module.
    *   `tests/output/`: Directory where test output images are saved (gitignored).
*   `requirements.txt`: Lists Python package dependencies.
*   `README.md`: This file.

## Sample Images in `data/`

The `data/` directory includes some sample images you can use to test the tool:

*   `sample_ref.png`: A basic reference image with a black square.
*   `sample_text.png`: An image containing the text "Hello World".
*   `sample_test_layout_issue.png`: A variation of `sample_ref.png` where the black square is moved, demonstrating layout shift detection.
*   `sample_test_missing_element.png`: A variation of `sample_ref.png` where the black square is missing.
*   `sample_test_color_issue.png`: A variation of `sample_ref.png` where the black square has a different color.

These allow you to quickly see the tool in action with predefined scenarios. For example:
```bash
python src/main.py \
  --ref_image data/sample_ref.png \
  --test_image data/sample_test_missing_element.png \
  --output_image output/result_missing.png
```

## Running Tests

To run the unit tests for this project:

1.  Ensure you have installed all dependencies, including those for testing (if any were separate - currently they are part of the main `requirements.txt`).
2.  Navigate to the root directory of the project.
3.  Run the following command:

    ```bash
    python -m unittest discover -s tests -v
    ```
    This will automatically discover and run all tests within the `tests` directory. Test output images are saved in `tests/output/`.

## Contributing (Placeholder)

Details on how to contribute to the project will be added here. For now, feel free to fork the repository and submit pull requests.

## License (Placeholder)

This project is currently unlicensed. A suitable open-source license will be added in the future.
```

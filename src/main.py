import argparse
import os
import logging

from PIL import Image
import cv2 # For cv2.boundingRect and other cv2 utilities if needed
import numpy as np # For contour processing if needed

# Project modules
from src import image_loader as il
from src import image_processor as ip
from src import visual_differ as vd
from src import text_analyzer as ta
from src import error_marker as em
from src import gemini_client as gc

# 1. Setup Logging (at the beginning of the script)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)

# Define a threshold for SSIM score
SSIM_THRESHOLD = 0.95
# Define a threshold for ORB matches
ORB_MIN_MATCHES_THRESHOLD = 10 # If fewer good matches, consider it a potential structural change
# Define a confidence threshold for OCR text to be considered for Gemini analysis or flagged as low quality
OCR_CONFIDENCE_THRESHOLD_FOR_GEMINI = 60
OCR_CONFIDENCE_THRESHOLD_FOR_LOW_QUALITY_FLAG = 50


def _cv_bbox_to_xyxy(cv_bbox: tuple) -> tuple[int, int, int, int]:
    """Converts OpenCV bbox (x,y,w,h) to (x1,y1,x2,y2)."""
    x, y, w, h = cv_bbox
    return int(x), int(y), int(x + w), int(y + h)

def process_images(ref_image_path: str, test_image_path: str, output_path: str, gemini_api_key: str | None = None):
    """
    Main processing pipeline for comparing reference and test images.
    """
    logging.info(f"Starting image comparison process for ref: '{ref_image_path}' and test: '{test_image_path}'")

    # a. Configure Gemini (Optional)
    if gemini_api_key:
        logging.info("Attempting to configure Gemini with provided API key.")
        gc.configure_gemini(api_key=gemini_api_key)
    else:
        # Attempt configuration with environment variable if no key is passed
        logging.info("No Gemini API key provided directly, attempting configuration via environment variable.")
        gc.configure_gemini() 

    if gc._gemini_configured: # Check internal flag from gemini_client
        logging.info("Gemini configured successfully.")
    else:
        logging.warning("Gemini not configured. Text coherence analysis via Gemini will be skipped.")

    # b. Load Images
    ref_img_pil = il.load_image(ref_image_path)
    test_img_pil = il.load_image(test_image_path)

    if ref_img_pil is None or test_img_pil is None:
        logging.error("Failed to load one or both images. Aborting.")
        return

    logging.info(f"Reference image loaded: {ref_img_pil.size}, Mode: {ref_img_pil.mode}")
    logging.info(f"Test image loaded: {test_img_pil.size}, Mode: {test_img_pil.mode}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")


    # c. Initialize Error List
    detected_errors: list[dict] = []

    # d. Preprocessing (Optional but Recommended)
    # For many visual comparisons, images should ideally be of the same size.
    # visual_differ functions handle some resizing internally, but explicit resizing
    # can be done here if a consistent size is desired for all checks.
    # For now, relying on internal resizing in vd functions.
    # Example:
    # max_dim = max(ref_img_pil.width, ref_img_pil.height, test_img_pil.width, test_img_pil.height, 1024)
    # ref_img_resized = ip.resize_image(ref_img_pil, max_dim)
    # test_img_resized = ip.resize_image(test_img_pil, max_dim)
    # Using original images for now, vd will handle internal alignment.

    # e. Visual Difference - SSIM
    logging.info("Performing SSIM comparison...")
    # Using copies for operations that might alter images (like convert) if not done by vd.
    # vd.compare_ssim expects PIL images and handles grayscale conversion internally.
    ssim_score, ssim_diff_image_cv = vd.compare_ssim(ref_img_pil, test_img_pil)
    if ssim_score is not None:
        logging.info(f"SSIM score: {ssim_score:.4f}")
        if ssim_score < SSIM_THRESHOLD:
            description = f'Low SSIM score: {ssim_score:.4f} (Threshold: {SSIM_THRESHOLD})'
            if ssim_diff_image_cv is not None:
                # The ssim_diff_image_cv is a visual representation.
                # To find specific differing regions, find contours in this diff image.
                # Threshold the SSIM diff image to get binary regions of difference
                # The ssim_diff_image is already uint8 in range [0,255]
                _, thresh_diff = cv2.threshold(ssim_diff_image_cv, 60, 255, cv2.THRESH_BINARY) # Threshold can be tuned
                contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    logging.info(f"Found {len(contours)} differing regions from SSIM diff image.")
                    for i, contour in enumerate(contours):
                        # Filter small contours
                        if cv2.contourArea(contour) > 50: # Min area threshold
                            bbox_cv = cv2.boundingRect(contour)
                            bbox_xyxy = _cv_bbox_to_xyxy(bbox_cv)
                            detected_errors.append({
                                'type': 'visual_discrepancy', 
                                'bbox': bbox_xyxy, 
                                'description': f'SSIM diff region {i+1} (Score: {ssim_score:.4f})', 
                                'source': 'SSIM'
                            })
                else: # If score is low but no specific contours found in diff, flag whole image
                    detected_errors.append({
                        'type': 'visual_discrepancy',
                        'bbox': (0, 0, test_img_pil.width, test_img_pil.height),
                        'description': description,
                        'source': 'SSIM (Overall)'
                    })

                # Save the SSIM difference image for debugging
                ssim_diff_pil = vd._cv2_to_pil(ssim_diff_image_cv)
                if ssim_diff_pil:
                    ssim_diff_path = os.path.join(output_dir, "debug_ssim_difference.png")
                    ssim_diff_pil.save(ssim_diff_path)
                    logging.info(f"Saved SSIM difference image to {ssim_diff_path}")
            else: # Score low, but no diff image returned
                 detected_errors.append({
                    'type': 'visual_discrepancy',
                    'bbox': (0, 0, test_img_pil.width, test_img_pil.height),
                    'description': description,
                    'source': 'SSIM (Overall)'
                })
    else:
        logging.warning("SSIM calculation failed.")

    # f. Visual Difference - ORB Features
    logging.info("Performing ORB feature comparison...")
    # vd.compare_orb_features expects PIL images.
    orb_matches, orb_visualization_cv = vd.compare_orb_features(ref_img_pil, test_img_pil, min_matches_threshold=ORB_MIN_MATCHES_THRESHOLD)
    if orb_matches is None or len(orb_matches) < ORB_MIN_MATCHES_THRESHOLD:
        logging.warning(f"Few ORB matches found (count: {len(orb_matches) if orb_matches else 0}). Potential structural differences.")
        detected_errors.append({
            'type': 'structural_change', 
            'bbox': (0, 0, test_img_pil.width, test_img_pil.height), 
            'description': f'Potential structural change detected. ORB matches: {len(orb_matches) if orb_matches else 0} (Threshold: {ORB_MIN_MATCHES_THRESHOLD}).', 
            'source': 'ORB'
        })
    if orb_visualization_cv is not None:
        orb_viz_pil = vd._cv2_to_pil(orb_visualization_cv)
        if orb_viz_pil:
            orb_viz_path = os.path.join(output_dir, "debug_orb_matches.png")
            orb_viz_pil.save(orb_viz_path)
            logging.info(f"Saved ORB feature match visualization to {orb_viz_path}")

    # g. Visual Difference - Contours
    logging.info("Performing contour comparison...")
    # vd.compare_contours expects PIL images.
    # The last element returned is now diff_contours_list
    _, _, diff_contour_img_cv, diff_contours_list = vd.compare_contours(ref_img_pil, test_img_pil) 
    
    if diff_contours_list:
        logging.info(f"Found {len(diff_contours_list)} difference contours.")
        for i, contour in enumerate(diff_contours_list):
            if cv2.contourArea(contour) > 30: # Min area threshold
                bbox_cv = cv2.boundingRect(contour)
                bbox_xyxy = _cv_bbox_to_xyxy(bbox_cv)
                detected_errors.append({
                    'type': 'layout_shift', 
                    'bbox': bbox_xyxy, 
                    'description': f'Contour difference region {i+1}.', 
                    'source': 'ContourDiff'
                })
    else:
        logging.info("No significant difference contours found.")

    if diff_contour_img_cv is not None:
        contour_diff_pil = vd._cv2_to_pil(diff_contour_img_cv)
        if contour_diff_pil:
            contour_diff_path = os.path.join(output_dir, "debug_contour_difference.png")
            contour_diff_pil.save(contour_diff_path)
            logging.info(f"Saved contour difference image to {contour_diff_path}")
            
    # h. Visual Difference - Color
    logging.info("Performing average color comparison...")
    # vd.compare_average_colors expects PIL images.
    similar_color, avg_color_ref, avg_color_test = vd.compare_average_colors(ref_img_pil, test_img_pil, tolerance=0.15) # Tolerance 0.15 = 15%
    if not similar_color:
        desc = f'Average color mismatch detected. Ref avg: {avg_color_ref}, Test avg: {avg_color_test}'
        logging.warning(desc)
        detected_errors.append({
            'type': 'color_change', 
            'bbox': (0, 0, test_img_pil.width, test_img_pil.height), 
            'description': desc, 
            'source': 'AvgColor'
        })
    else:
        logging.info(f"Average colors are similar. Ref: {avg_color_ref}, Test: {avg_color_test}")

    # i. Text Extraction and Analysis (on Test Image)
    logging.info("Performing text extraction and analysis on test image...")
    # ta.extract_text_from_image expects a PIL image.
    text_data_test = ta.extract_text_from_image(test_img_pil)
    
    if text_data_test:
        logging.info(f"Extracted {len(text_data_test)} text segments from test image.")
        # Analyze text coherence if Gemini is configured
        # analyze_text_coherence now only takes text_data
        analyzed_text_data_test = ta.analyze_text_coherence(text_data_test)
        
        for item in analyzed_text_data_test:
            # Bbox from Tesseract: item['left'], item['top'], item['width'], item['height']
            bbox_cv = (item['left'], item['top'], item['width'], item['height'])
            bbox_xyxy = _cv_bbox_to_xyxy(bbox_cv)
            text_content = item.get('text', '')
            confidence = item.get('conf', 0.0)
            gemini_analysis_result = item.get('gemini_analysis', '')

            # Check Gemini analysis first
            if gemini_analysis_result and "Analysis skipped" not in gemini_analysis_result and "No text to analyze" not in gemini_analysis_result and "Gemini analysis failed" not in gemini_analysis_result and "No specific feedback" not in gemini_analysis_result:
                # Heuristic: if Gemini response is not one of the generic "OK" or "skipped" messages, consider it an issue.
                # A more robust check would be to parse Gemini's output for specific keywords or structure.
                is_issue_by_gemini = True 
                # Example: if "nonsensical" in gemini_analysis_result.lower() or "error" in gemini_analysis_result.lower():
                # For now, any detailed feedback from Gemini is treated as an issue.
                if gemini_analysis_result.startswith("Gemini client not configured") or gemini_analysis_result == "Not analyzed (low confidence or no text).":
                    is_issue_by_gemini = False

                if is_issue_by_gemini:
                    detected_errors.append({
                        'type': 'text_issue', 
                        'bbox': bbox_xyxy, 
                        'description': f'Text: "{text_content}" - Gemini: {gemini_analysis_result}', 
                        'source': 'GeminiTextAnalysis'
                    })
            # Then check for low confidence text not already flagged by Gemini
            elif confidence < OCR_CONFIDENCE_THRESHOLD_FOR_LOW_QUALITY_FLAG:
                detected_errors.append({
                    'type': 'text_quality', 
                    'bbox': bbox_xyxy, 
                    'description': f'Low confidence text: "{text_content}" (Conf: {confidence:.2f})', 
                    'source': 'OCRConfidence'
                })
    else:
        logging.info("No text extracted from test image.")

    # j. Mark Errors
    logging.info(f"Total errors/observations detected: {len(detected_errors)}")
    if detected_errors:
        logging.info("Marking detected errors on the test image.")
        # em.mark_errors_on_image expects a PIL image.
        marked_image_pil = em.mark_errors_on_image(test_img_pil, detected_errors)
        if marked_image_pil:
            try:
                marked_image_pil.save(output_path)
                logging.info(f"Successfully saved marked image to: {output_path}")
            except Exception as e:
                logging.error(f"Failed to save marked image to {output_path}: {e}")
        else:
            logging.error("Failed to generate marked image.")
    else:
        logging.info("No significant errors detected to mark.")
        # Optionally save the original test image to output_path or do nothing.
        try:
            # For consistency, save the test image even if no errors.
            test_img_pil.save(output_path) 
            logging.info(f"No errors detected. Saved original test image to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save original test image to {output_path}: {e}")


    # k. (Optional) Text Report
    if detected_errors:
        print("\n--- Summary of Detected Errors/Observations ---")
        for i, error in enumerate(detected_errors):
            print(f"{i+1}. Type: {error.get('type', 'N/A')}, Source: {error.get('source', 'N/A')}")
            print(f"   BBox: {error.get('bbox', 'N/A')}")
            print(f"   Desc: {error.get('description', 'N/A')}")
        print("---------------------------------------------\n")
    else:
        print("\n--- No significant errors or observations detected. ---\n")
    
    logging.info("Image comparison process finished.")


# 4. main Function (CLI Parsing)
def main():
    """
    Parses command-line arguments and initiates the image processing.
    """
    parser = argparse.ArgumentParser(description="Compare two images and detect differences.")
    parser.add_argument("--ref_image", required=True, help="Path to the reference image file.")
    parser.add_argument("--test_image", required=True, help="Path to the test image file.")
    parser.add_argument("--output_image", required=True, help="Path to save the output image with marked errors.")
    parser.add_argument("--gemini_api_key", required=False, default=None, help="Optional Google Gemini API key.")
    
    args = parser.parse_args()

    # Basic input validation
    if not os.path.exists(args.ref_image):
        logging.error(f"Reference image path does not exist: {args.ref_image}")
        return
    if not os.path.exists(args.test_image):
        logging.error(f"Test image path does not exist: {args.test_image}")
        return

    process_images(args.ref_image, args.test_image, args.output_image, args.gemini_api_key)

# 5. if __name__ == "__main__": Block
if __name__ == "__main__":
    main()
```

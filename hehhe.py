# screen_text_extractor.py
#
# Installation instructions:
# 1. Install Python libraries: pip install pytesseract pillow pyautogui opencv-python
# 2. Install Tesseract OCR (required for text extraction):
#    - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.
#    - Linux: sudo apt install tesseract-ocr
#    - Mac: brew install tesseract
#    Ensure Tesseract is in your system PATH.

import argparse
import datetime
import threading
import time
import sys

import pyautogui
from PIL import Image
import cv2
import numpy as np
import pytesseract
from pytesseract import pytesseract as pt  # For exception checking

# Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(image: Image.Image) -> str:
    """
    Extracts text from a PIL Image using OCR with pre-processing for optimization.
    Converts to grayscale, applies Otsu's thresholding, then runs Tesseract.
    """
    try:
        # Convert PIL Image to OpenCV format
        img_np = np.array(image)
        # Pre-process: grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # Thresholding for better OCR accuracy
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Run OCR
        text = pytesseract.image_to_string(thresh)
        if not text.strip():
            return "No text detected"
        return text
    except pt.TesseractNotFoundError:
        print("Error: Tesseract not found. Please install Tesseract OCR and ensure it's in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return "No text detected"

def append_to_file(text: str) -> None:
    """
    Appends extracted text to Screen_etxt.txt with a timestamp.
    """
    timestamp = datetime.datetime.now().isoformat()
    with open("Screen_etxt.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]: {text}\n\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Screen Text Extractor")
    parser.add_argument("--interval", type=float, default=2.0, help="Capture interval in seconds for screen-recording mode (default: 2.0)")
    args = parser.parse_args()

    # User selection
    mode = input("Type 'sh' for screenshot mode or 'sr' for screen-recording mode: ").strip().lower()

    if mode == 'sh':
        # Screenshot mode
        print("Waiting 3 seconds to arrange screen...")
        time.sleep(3)
        screenshot = pyautogui.screenshot()
        text = extract_text(screenshot)
        append_to_file(text)
        print("Screenshot captured and text extracted.")

    elif mode == 'sr':
        # Screen-recording mode (periodic screenshots)
        print(f"Starting screen recording every {args.interval} seconds. Type 'q' and press Enter to stop.")
        stop_event = threading.Event()

        def listen_for_q():
            while not stop_event.is_set():
                inp = input().strip().lower()
                if inp == 'q':
                    stop_event.set()
                    break

        # Start listener thread
        listener_thread = threading.Thread(target=listen_for_q, daemon=True)
        listener_thread.start()

        try:
            while not stop_event.is_set():
                screenshot = pyautogui.screenshot()
                text = extract_text(screenshot)
                append_to_file(text)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass  # Graceful exit on Ctrl+C
        finally:
            stop_event.set()  # Ensure stop
            listener_thread.join()
            print("Screen recording stopped.")

    else:
        print("Invalid mode selected. Exiting.")

if __name__ == "__main__":
    main()
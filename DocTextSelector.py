
# Enhanced Python App: DocTextSelector
# Prompts for input mode (camera, screen, terminal), shows formatted terminal output,
# saves terminal/OCR matches to text.txt, screen-detected text to screen_text.txt,
# and PDF matches to text2.txt in Markdown. Features: fullscreen camera, robust screen capture
# with mss and pywin32 fallback, sentence-based paragraph search, detailed error logging,
# improved exit handling with Ctrl+C and window close support.

import cv2
import mediapipe as mp
import pytesseract
import fitz  # PyMuPDF
import re
import numpy as np
import os
from mss import mss
from mss.exception import ScreenShotError
from typing import List, Tuple, Optional
import textwrap
import time
import logging
import signal
import sys

# Fallback screen capture with pywin32
try:
    import win32gui
    import win32ui
    import win32con
    PYWIN32_AVAILABLE = True
except ImportError:
    PYWIN32_AVAILABLE = False
    print("pywin32 not installed. Install with 'pip install pywin32' for fallback screen capture.")

# Setup logging
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Configuration
CAMERA_ID = 0
OCR_CONF_THRESHOLD = 40  # Adjustable for text detection
SEARCH_FILES = ['./semiconductor.pdf']  # Relative path
GESTURE_SELECT_DISTANCE = 35  # Selection radius
OVERLAY_ALPHA = 0.6  # Overlay transparency
DEBOUNCE_FRAMES = 10  # Debounce frames
DEFAULT_TEXT = ["MATERIALS", "INTRODUCTION"]  # Default sections for display
OCR_INTERVAL = 5  # Optimized OCR frequency
TERMINAL_WRAP_WIDTH = 80  # Word wrap width for terminal and file output
GUI_LINE_LENGTH = 80  # Max characters per line in GUI overlay
OUTPUT_FILE = 'text.txt'  # Terminal/OCR matches
SCREEN_TEXT_FILE = 'screen_text.txt'  # Screen-detected text
MATCHES_FILE = 'text2.txt'  # PDF matches
FALLBACK_REGION = {'top': 0, 'left': 0, 'width': 1280, 'height': 720}  # Fallback if full screen fails
CAPTURE_TIMEOUT = 5  # Seconds to wait for a capture before retrying

# Cleanup function
def cleanup(cap=None, hands=None, sct=None):
    logging.info("Cleaning up resources.")
    if cap and cap.isOpened():
        cap.release()
    if hands:
        hands.close()
    if sct:
        sct.close()
    cv2.destroyAllWindows()
    logging.info("Program terminated cleanly.")

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    print("\nCtrl+C received. Exiting program.")
    logging.info("Program terminated via Ctrl+C.")
    cleanup(cap, hands, sct)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Diagnostic: Verify Tesseract
try:
    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError(f"Tesseract not found at {TESSERACT_PATH}")
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
except Exception as e:
    logging.error(f"Tesseract error: {e}")
    print(f"Tesseract error: {e}")
    cleanup()
    exit(1)

# Preload files with sentence tracking
file_sentences = {}
for file_path in SEARCH_FILES:
    try:
        full_path = os.path.abspath(file_path)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found")
            continue
        if file_path.endswith('.pdf'):
            doc = fitz.open(full_path)
            sentences = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                # Split text into sentences using period as delimiter
                page_sentences = re.split(r'(?<=[.!?])\s+', text)
                page_sentences = [s.strip() for s in page_sentences if s.strip()]
                for sentence in page_sentences:
                    sentences.append((page_num + 1, sentence))
            doc.close()
        file_sentences[file_path] = sentences
    except Exception as e:
        logging.error(f"File error {file_path}: {e}")
        print(f"File error {file_path}: {e}")

# MediaPipe and MSS init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
sct = mss()

# Get full screen dimensions with error handling
try:
    with sct:
        monitor = sct.monitors[0]  # Primary monitor
        SCREEN_REGION = {
            'top': monitor['top'],
            'left': monitor['left'],
            'width': monitor['width'],
            'height': monitor['height']
        }
        print(f"Full screen region: {SCREEN_REGION}")
except Exception as e:
    logging.error(f"Failed to get full screen dimensions: {e}")
    print(f"Failed to get full screen dimensions: {e}. Using fallback region.")
    SCREEN_REGION = FALLBACK_REGION

# Fallback screen capture using pywin32
def capture_screen_pywin32():
    if not PYWIN32_AVAILABLE:
        raise ScreenShotError("pywin32 not available for fallback capture.")
    try:
        hdesktop = win32gui.GetDesktopWindow()
        width = SCREEN_REGION['width']
        height = SCREEN_REGION['height']
        left = SCREEN_REGION['left']
        top = SCREEN_REGION['top']

        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc = img_dc.CreateCompatibleDC()

        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot)
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)

        bmp_info = screenshot.GetInfo()
        bmp_str = screenshot.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype=np.uint8).reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        mem_dc.DeleteDC()
        img_dc.DeleteDC()
        win32gui.ReleaseDC(hdesktop, desktop_dc)
        win32gui.DeleteObject(screenshot.GetHandle())
        return img
    except Exception as e:
        logging.error(f"pywin32 screen capture failed: {e}")
        raise ScreenShotError(f"pywin32 screen capture failed: {e}")

# Enhanced OCR with dynamic preprocessing
def ocr_with_boxes(frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)  # Noise reduction
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESHOLD_BINARY, 11, 2)
        if np.mean(thresh) < 100:  # Fallback for dark images
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config='--psm 6')
        words = []
        for i in range(len(data['level'])):
            conf = int(data['conf'][i])
            if conf > OCR_CONF_THRESHOLD and data['text'][i].strip():
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                words.append((data['text'][i].lower(), (x, y, x + w, y + h), conf))
        return words
    except Exception as e:
        logging.error(f"OCR error: {e}")
        print(f"OCR error: {e}")
        return []

# Save detected screen text to file
def save_screen_text(words: List[Tuple[str, Tuple[int, int, int, int], float]]):
    if not words:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SCREEN_TEXT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"## OCR Capture at {timestamp}\n\n")
        f.write("### Detected Text:\n\n")
        for i, (word, _, conf) in enumerate(words, 1):
            wrapped = textwrap.fill(word, width=TERMINAL_WRAP_WIDTH)
            f.write(f"{i}. **Word**: {wrapped} (Confidence: {conf:.0f}%)\n\n")
        f.write("===" + "=" * (TERMINAL_WRAP_WIDTH - 3) + "\n\n")

# Closest word finder
def find_closest_word(point: Tuple[int, int], words: List[Tuple[str, Tuple[int, int, int, int], float]]) -> Optional[str]:
    min_dist = float('inf')
    selected = None
    for word, (x1, y1, x2, y2), _ in words:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        dist = ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
        if dist < min_dist and dist < GESTURE_SELECT_DISTANCE:
            min_dist = dist
            selected = word
        if min_dist == 0:
            break
    return selected

# Exact word search with sentence-based highlighting and source tracking
def search_in_files(word: str) -> List[Tuple[str, str]]:  # Returns list of (source, highlighted_sentence)
    word_lower = word.lower()
    matches = []
    for file_path, sentences in file_sentences.items():
        file_name = os.path.basename(file_path)
        for page, sentence in sentences:
            sentence_lower = sentence.lower()
            if re.search(r'\b' + re.escape(word_lower) + r'\b', sentence_lower):
                highlighted = re.sub(r'\b(' + re.escape(word_lower) + r')\b', r'**\1**', sentence, flags=re.IGNORECASE)
                source = f"{file_name} (Page {page})"
                matches.append((source, highlighted.strip()))
    return matches[:3]  # Limit to top 3 matches

# Format and save terminal/OCR matches
def save_to_file(word: str, matches: List[Tuple[str, str]], file_path: str = OUTPUT_FILE):
    with open(file_path, 'a', encoding='utf-8') as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"## Search for: {word} (at {timestamp})\n\n")
        if matches:
            f.write("### Matches Found:\n\n")
            for i, (source, match) in enumerate(matches, 1):
                wrapped = textwrap.fill(match, width=TERMINAL_WRAP_WIDTH)
                f.write(f"{i}. **Source**: {source}\n\n")
                f.write(f"{wrapped}\n\n")
                f.write("-" * TERMINAL_WRAP_WIDTH + "\n\n")
        else:
            f.write("No matches found.\n\n")
        f.write("===" + "=" * (TERMINAL_WRAP_WIDTH - 3) + "\n\n")

# Function to wrap text for GUI overlay
def wrap_text_for_gui(text: str, max_length: int = GUI_LINE_LENGTH) -> List[str]:
    wrapped = []
    for line in text.split('\n'):
        wrapped.extend(textwrap.wrap(line, width=max_length))
    return wrapped

# Main loop setup
mode = input("Choose input mode (c for camera, s for screen, t for terminal): ").lower()
use_camera = mode == 'c'
use_screen = mode == 's'
use_terminal = mode == 't'

if use_camera:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Camera error, switching to screen mode.")
        use_camera = False
        use_screen = True
else:
    cap = None

# Set up fullscreen window
window_name = 'DocTextSelector'
if not use_terminal:
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

selected_word = None
matches = []
ocr_words = []
frame_count = 0
debounce_count = 0
prev_selected = None
use_pywin32 = False  # Flag for pywin32 fallback
last_capture_time = time.time()

# Clear output files at start
for file in [OUTPUT_FILE, SCREEN_TEXT_FILE, MATCHES_FILE]:
    if os.path.exists(file):
        os.remove(file)

try:
    while True:
        if use_terminal:
            selected_word = input("Enter word to search (or 'q' to quit): ").lower()
            if selected_word == 'q':
                break
            matches = search_in_files(selected_word)
            # Display results in terminal with formatting and word wrap
            print(f"\n## Searching for: {selected_word}\n")
            if matches:
                print("### Matches Found:\n")
                for i, (source, match) in enumerate(matches, 1):
                    print(f"{i}. **Source**: {source}")
                    wrapped = textwrap.fill(match, width=TERMINAL_WRAP_WIDTH)
                    print(f"{wrapped}\n")
                    print("-" * TERMINAL_WRAP_WIDTH + "\n")
            else:
                print("No matches found.\n")
            print("=" * TERMINAL_WRAP_WIDTH + "\n")
            # Save to text.txt and text2.txt
            save_to_file(selected_word, matches, OUTPUT_FILE)
            save_to_file(selected_word, matches, MATCHES_FILE)
        else:
            # Capture frame
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame error, switching to screen mode.")
                    use_camera = False
                    use_screen = True
                    continue
                frame = cv2.flip(frame, 1)  # Mirror flip for camera only
            else:  # use_screen
                if time.time() - last_capture_time > CAPTURE_TIMEOUT:
                    logging.error("Screen capture timeout. Switching to camera mode.")
                    print("Screen capture timeout. Switching to camera mode.")
                    use_camera = True
                    use_screen = False
                    cap = cv2.VideoCapture(CAMERA_ID)
                    if not cap.isOpened():
                        print("Camera also unavailable. Exiting.")
                        break
                    continue
                try:
                    if use_pywin32:
                        frame = capture_screen_pywin32()
                    else:
                        screenshot = np.array(sct.grab(SCREEN_REGION))
                        frame = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2RGB)
                    last_capture_time = time.time()
                except ScreenShotError as e:
                    logging.error(f"Screen capture error: {e}")
                    print(f"Screen capture error: {e}. Switching to fallback region.")
                    SCREEN_REGION = FALLBACK_REGION
                    try:
                        screenshot = np.array(sct.grab(SCREEN_REGION))
                        frame = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2RGB)
                        last_capture_time = time.time()
                    except ScreenShotError as e:
                        logging.error(f"Fallback screen capture failed: {e}")
                        print(f"Fallback screen capture failed: {e}.")
                        if PYWIN32_AVAILABLE:
                            print("Attempting pywin32 fallback capture.")
                            use_pywin32 = True
                            try:
                                frame = capture_screen_pywin32()
                                last_capture_time = time.time()
                            except ScreenShotError as e:
                                logging.error(f"pywin32 fallback failed: {e}")
                                print(f"pywin32 fallback failed: {e}. Switching to camera mode.")
                                use_camera = True
                                use_screen = False
                                cap = cv2.VideoCapture(CAMERA_ID)
                                if not cap.isOpened():
                                    print("Camera also unavailable. Exiting.")
                                    break
                                continue
                        else:
                            print("No pywin32 available. Switching to camera mode.")
                            use_camera = True
                            use_screen = False
                            cap = cv2.VideoCapture(CAMERA_ID)
                            if not cap.isOpened():
                                print("Camera also unavailable. Exiting.")
                                break
                            continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand detection
            results = hands.process(rgb_frame)
            finger_tip = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    finger_tip = (int(tip.x * w), int(tip.y * h))
                    cv2.circle(frame, finger_tip, 8, (0, 255, 0), -1)

            # OCR at optimized interval
            if frame_count % OCR_INTERVAL == 0:
                ocr_words = ocr_with_boxes(frame)
                if use_screen and ocr_words:
                    save_screen_text(ocr_words)

            # Draw word boxes with confidence
            for word, (x1, y1, x2, y2), conf in ocr_words:
                color = (0, 0, 255) if word == selected_word else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{conf:.0f}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Gesture selection with debounce
            if finger_tip and ocr_words:
                new_selected = find_closest_word(finger_tip, ocr_words)
                if new_selected == prev_selected:
                    debounce_count += 1
                else:
                    debounce_count = 0
                    prev_selected = new_selected
                if debounce_count >= DEBOUNCE_FRAMES and new_selected != selected_word:
                    selected_word = new_selected
                    matches = search_in_files(selected_word)
                    # Save OCR matches to text.txt and text2.txt
                    save_to_file(selected_word, matches, OUTPUT_FILE)
                    save_to_file(selected_word, matches, MATCHES_FILE)
                    debounce_count = 0

            # Overlay matches and default text (for camera/screen only)
            overlay = frame.copy()
            y_offset = 50
            # Display default text
            for text in DEFAULT_TEXT:
                cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            # Display selected word
            if selected_word:
                cv2.putText(overlay, f"Selected: {selected_word}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_offset += 30
            # Display matches with source and wrapped lines
            for source, match in matches:
                cv2.putText(overlay, f"Source: {source}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                y_offset += 30
                wrapped_lines = wrap_text_for_gui(match)
                for line in wrapped_lines[:6]:  # Limit lines to avoid overflow
                    cv2.putText(overlay, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    y_offset += 25
                y_offset += 15
            cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

            cv2.imshow(window_name, frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                use_camera = False
                use_screen = True
                use_pywin32 = False
                print("Switched to screen mode (mss).")
            elif key == ord('c'):
                if not use_camera:
                    cap = cv2.VideoCapture(CAMERA_ID)
                    if cap.isOpened():
                        use_camera = True
                        use_screen = False
                        print("Switched to camera mode.")
                    else:
                        print("Camera unavailable, staying in screen mode.")
            elif key == ord('q'):
                print("Exiting via 'q' key.")
                break

except KeyboardInterrupt:
    print("\nCtrl+C received. Exiting program.")
    logging.info("Program terminated via Ctrl+C.")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    print(f"Unexpected error: {e}. Check error_log.txt for details.")
finally:
    cleanup(cap, hands, sct)

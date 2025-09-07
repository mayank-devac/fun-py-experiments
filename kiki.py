import pyautogui
import cv2
import numpy as np
import pytesseract
import textwrap
import datetime
import sys
import threading
import os

# Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ------------- CONFIG -------------
OUTPUT_TEXT_FILE = "Screen_etxt.txt"
OUTPUT_VIDEO_FILE = "screenvideo.mp4"
CAPTURE_INTERVAL = 2  # seconds between OCR frames
OCR_LANGS = 'eng+hin'  # OCR languages
# ----------------------------------

def format_text(text: str) -> str:
    """Clean, wrap, and format OCR text into Markdown bullet list."""
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.endswith('.'):
            line += '.'
        wrapped = textwrap.fill(line, width=80)
        formatted_lines.append(f"- {wrapped}")
    if not formatted_lines:
        formatted_lines.append("- [No text detected].")
    return "\n".join(formatted_lines)

def save_text(extracted: str):
    """Save extracted text with timestamp and markdown formatting."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## 🕒 {timestamp}\n\n{extracted}\n\n---\n"
    with open(OUTPUT_TEXT_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

def ocr_from_image(img) -> str:
    """Run Tesseract OCR on a numpy image array."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang=OCR_LANGS)
    return text

def screenshot_mode():
    """Take a single screenshot after 3-second delay and save text."""
    print("Taking screenshot in 3 seconds…")
    pyautogui.sleep(3)
    img = pyautogui.screenshot()
    text = ocr_from_image(img)
    save_text(format_text(text))
    print("Saved to", OUTPUT_TEXT_FILE)

def recording_mode(interval: int = CAPTURE_INTERVAL):
    """
    Continuously capture screen frames and save a video
    plus extract text every interval seconds until user presses 'q' + Enter.
    """
    print("Recording… press 'q' + Enter to stop.")
    stop_flag = False

    def listen_for_q():
        nonlocal stop_flag
        while True:
            if sys.stdin.readline().strip().lower() == 'q':
                stop_flag = True
                break

    threading.Thread(target=listen_for_q, daemon=True).start()

    # Set up VideoWriter
    # Get screen size
    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_FILE,
        fourcc,
        10.0,  # fps
        (screen_size.width, screen_size.height)
    )

    last_ocr_time = datetime.datetime.now()

    while not stop_flag:
        # Capture screen
        img = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(frame)

        # Every interval seconds, run OCR
        now = datetime.datetime.now()
        if (now - last_ocr_time).total_seconds() >= interval:
            text = ocr_from_image(img)
            save_text(format_text(text))
            last_ocr_time = now

        # Aim for ~10fps recording
        cv2.waitKey(100)  # 0.1s pause for ~10fps

    video_writer.release()
    print("Recording stopped. Video saved to", OUTPUT_VIDEO_FILE)
    print("Text saved to", OUTPUT_TEXT_FILE)

if __name__ == "__main__":
    mode = input("Type 'sh' for screenshot, 'sr' for screenrecording: ").strip().lower()
    if mode == 'sh':
        screenshot_mode()
    elif mode == 'sr':
        recording_mode()
    else:
        print("Unknown mode.")

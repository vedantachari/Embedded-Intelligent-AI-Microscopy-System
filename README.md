# YOLO Dashboard (Kivy) — README

A small Kivy-based desktop/Raspberry Pi app that runs a YOLO model on a video (or camera / Pi Camera), shows a live dashboard of detected classes and **unique counts**, allows saving a CSV of counts, writes an output video, and can fetch detailed class info via the Gemini API.  

- GUI / main logic: `main.py`. :contentReference[oaicite:0]{index=0}  
- Info popup (UI + clipboard + background fetch): `info_popup.py`. :contentReference[oaicite:1]{index=1}  
- Gemini API helper (wraps Google GenAI client and parses JSON output): `gemini_client.py`. :contentReference[oaicite:2]{index=2}

---

## Features (quick)

- Load a YOLO model (`.pt`) and run inference on:
  - Local video file (mp4/avi/...)
  - Webcam (`0`)
  - Pi Camera (if `picamera2` is installed and available)
- Live display of the video with bounding boxes and per-class counts.
- Simple tracking to count **unique** objects over time.
- Save unique counts as `dashboard_counts.csv`.
- Save annotated output to `output_dashboard.mp4`.
- "Info" button on each class that queries Gemini and displays structured info in a popup (requires API key).

---

## Prerequisites

- Python 3.9+ recommended.
- A modern CPU; GPU is optional (if your PyTorch/Ultralytics is GPU-enabled).
- `ffmpeg` (optional but useful if you work with some video codecs).
- If running on Raspberry Pi and you want Pi camera support: `picamera2` + required Pi libs.

---

## Recommended Python dependencies

Create a `requirements.txt` (example below) and install into a venv:

```text
kivy>=2.1.0
opencv-python
ultralytics>=8.0.0
google-genai
numpy
```
Install with:
```code
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
### How to run
 - Activate your venv (if used).
 - Ensure dependencies are installed and (optionally) GOOGLE_API_KEY is set.
 - Start the app:
     python main.py

The Kivy GUI will open:
 - Click Load Model → choose a .pt model (YOLO format).
 - Click Load Video → choose Video File, then pick /mnt/data/hello.mp4 (or select Camera for webcam / 0 for default).
 - Click Start to begin processing. Click Stop to end.
 - Click Save CSV to export dashboard_counts.csv with columns class_index,class_name,unique_count.
 - Use the small i info button next to a class to open the Gemini-powered info popup (requires API key). See info_popup.py for how the popup is implemented. 

# AI-Powered Microscopy Assistant 

**Real-time classification, counting, and biological analysis of marine organisms using Raspberry Pi/Jetson, YOLOv8, and Kivy.**

## Project Overview

This project converts a standard optical microscope into an intelligent research tool. Using a 3D-printed mount system and an embedded AI computer (Raspberry Pi 5 or Jetson), the system performs **real-time detection and counting** of marine organisms (like diatoms and plankton) directly from the microscope's eyepiece feed.

Beyond simple detection, the application integrates **Google's Gemini API** to provide instant, detailed biological context for identified species, assisting researchers and students in taxonomy and identification.

### Key Features
* **Real-Time Detection:** Powered by **YOLOv8** for fast and accurate organism recognition.
* **Unique Counting:** Implements an IOU-based object tracker to count *unique* organisms as they pass through the frame, rather than per-frame detections.
* **Touchscreen Interface:** A modern, touch-friendly UI built with **Kivy**, designed for portable screens.
* **Generative AI Insights:** Click on any detected class to fetch a detailed "Biologist Report" (taxonomy, identification tips, size) generated dynamically by **Google Gemini**.
* **Data Export:** Save session statistics to CSV for further analysis.
* **Hardware Flexible:** Supports standard Webcams, Video Files (`.mp4`), and native **Raspberry Pi Camera (Libcamera/Picamera2)**.

---

![Image](https://github.com/user-attachments/assets/ec3777dc-6b2a-4c71-9310-4d086224239b)

## Hardware Setup

The physical system is designed to be a "plug-and-play" accessory for existing microscopes.

* **Compute:** Raspberry Pi 5 or NVIDIA Jetson Nano.
* **Imaging:** Raspberry Pi Camera Module or USB Microscope Camera.
* **Display:** Touchscreen monitor (mounted to the rig).
* **Mounting:** Custom 3D-printed parts designed in CATIA.
    * *Camera Holder:* Fits standard microscope eyepieces.
    * *Display Holder:* Mounts the screen to the microscope arm.
    
<img width="1110" height="688" alt="Image" src="https://github.com/user-attachments/assets/fa9c7985-c2e8-461c-ac2c-54928452bbd5" />

![Image](https://github.com/user-attachments/assets/458d5b53-d2cd-4ae5-a48b-8e12029c9f32)

---

## Software Architecture

The software is written in Python and follows this processing pipeline:

1.  **Input:** Capture microscopic image stream (Picamera2 or OpenCV).
2.  **Preprocessing:** Noise removal and formatting.
3.  **Inference:** `Ultralytics YOLOv8` model predicts bounding boxes and classes.
4.  **Tracking:** `SimpleTracker` associates detections across frames to maintain unique IDs.
5.  **Interface:** `Kivy` renders the video feed, bounding boxes, and statistics.
6.  **Analysis:** On-demand API calls to `Gemini` for biological descriptions.

### File Structure
* `main.py`: The entry point. Handles the Kivy UI, video threads, and YOLO inference.
* `gemini_client.py`: Wrapper for Google GenAI SDK to fetch biological data.
* `info_popup.py`: UI helper for displaying the AI-generated biological info.

---


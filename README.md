# Raspberry Pi Smart Car — Smart Control & Object Detection

An embedded-systems course project combining **Raspberry Pi hardware control**, a **Flask-based web interface**, **live camera streaming**, and **TensorFlow object detection**. The goal was to explore hardware–software integration and support safer car operation through visual detection and alerts.

## Project overview

Developed for the **Embedded System Design and Practice** course, this prototype supports car movement through physical buttons and browser-based controls. A camera supplies video frames for object detection; the application can display a warning and trigger a buzzer when an object is detected in a specified area of the frame.

## Features

- **Directional movement:** forward, backward, left, and right, using Raspberry Pi GPIO motor-control signals.
- **Two control methods:** physical input buttons and a web interface with directional buttons.
- **Camera feed:** streams camera frames to a webpage using Flask and OpenCV.
- **Object detection:** uses a TensorFlow SSD Lite MobileNet model to detect objects in camera frames.
- **Warning feedback:** displays an alert in the web interface and drives a buzzer for selected detections.

## Technologies

| Area | Tools / technologies |
| --- | --- |
| Embedded hardware | Raspberry Pi, GPIO, DC motor control, input buttons, buzzer, camera |
| Backend | Python, Flask |
| Computer vision | OpenCV, TensorFlow Object Detection API, SSD Lite MobileNet V2 |
| Frontend | HTML, CSS |

## Project structure

```text
RaspberryPi-SmartCar/
├── app.py                 # Flask routes, movement logic, video feed, alert handling
├── camera.py              # OpenCV camera capture
├── object_detection.py    # Standalone object-detection script
├── templates/
│   ├── stream.html        # Browser-based movement controls and video display
│   └── alert.html         # Alert display
├── static/
│   └── style.css          # Web interface styling
├── RequiredReferences.txt # Original TensorFlow model setup notes
└── README.md
```

## System workflow

1. The user presses a physical button or sends a direction command from the Flask webpage.
2. The Raspberry Pi GPIO outputs control the car's motors.
3. The camera captures video frames, which are streamed to the webpage.
4. TensorFlow processes frames for object detection.
5. The application displays a warning and activates a buzzer when its detection conditions are met.

## Setup notes and limitations

This repository preserves the **project source code**, not a fully packaged or hardware-tested installation. Running it requires a compatible Raspberry Pi and GPIO wiring, Python packages, TensorFlow Object Detection API utilities, a COCO label map, and the `ssdlite_mobilenet_v2_coco_2018_05_09` model files. Those external model assets and utilities are **not included** in this repository.

The implementation uses legacy TensorFlow graph-loading APIs and Raspberry Pi-specific libraries, so dependency versions and hardware configuration need to be checked before running. The source code has not been verified here on the original hardware. Do not operate a moving car without appropriate supervision and safety precautions.

## Learning focus

- Embedded hardware and software integration
- GPIO-based motor and button control
- Flask web interfaces for device interaction
- Camera integration and real-time image processing
- Object detection and alert logic

## Portfolio context

This project appears as **Project 5: Smart Control** in my 2026 technical portfolio, under the course *Embedded System Design and Practice*.

**Author:** Phuong Dieu Thao - Jerry (芳妙草)

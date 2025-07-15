# Codebase Onboarding Guide

## 1. High-Level Goal

This codebase is a **Group Behavior Analysis System** designed to solve the problem of automated crowd and social interaction monitoring. The primary purpose is to:

- **Detect and track people** in video footage using computer vision (YOLO model)
- **Identify group formations** when 3+ people cluster together using spatial analysis
- **Calculate dwell times** to measure how long groups stay together
- **Generate detailed logs** of group formation, dispersal, and duration events
- **Provide real-time visual feedback** through a web-based interface

This system is particularly valuable for retail analytics, public space monitoring, social interaction studies, crowd behavior analysis, and security applications.

## 2. Architecture Overview

The system follows a **Monolithic Architecture** with a clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │  VideoProcessor  │    │    Output       │
│   Web Interface │───▶│  Core Logic      │───▶│    Files        │
│   (app.py)      │    │  (video_processor│    │  (CSV + Images) │
│                 │    │      .py)        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   AI Models     │              │
         │              │ • YOLO (detect) │              │
         │              │ • ByteTrack     │              │
         │              │ • DBSCAN        │              │
         │              └─────────────────┘              │
         │                                               │
         └───────────────────────────────────────────────┘
```

**Data Flow:** User uploads video → Streamlit interface → VideoProcessor processes frame-by-frame → AI models detect/track/cluster → Results saved to output directory → User views results and downloads logs.

## 3. File and Directory Breakdown

### Core Application Files
- **Path:** `app.py`
- **Purpose:** Streamlit web application serving as the main user interface. Handles file uploads, displays real-time processing progress, and provides download links for results.

- **Path:** `src/video_processor.py`
- **Purpose:** Contains the `VideoProcessor` class with core computer vision logic. Handles person detection, tracking, group identification, and event logging.

- **Path:** `src/__init__.py`
- **Purpose:** Empty Python package initializer making the `src` directory a proper Python module.

### Configuration and Dependencies
- **Path:** `requirements.txt`
- **Purpose:** Lists all Python dependencies needed to run the application (6 packages total).

- **Path:** `README.md`
- **Purpose:** Comprehensive documentation explaining the system's purpose, features, applications, and technical stack.

### AI Model
- **Path:** `yolov8n.pt`
- **Purpose:** Pre-trained YOLOv8 nano model file for person detection. This is the core object detection model.

### Data Directories
- **Path:** `output/`
- **Purpose:** Directory where processed results are saved including CSV logs and snapshot images.

- **Path:** `video/`
- **Purpose:** Directory intended for storing input video files.

### Testing
- **Path:** `test.py`
- **Purpose:** Currently empty test file - represents a testing framework placeholder.

## 4. Core Logic and Workflow

The application follows this primary workflow:

1. **Video Input:** User uploads video through Streamlit interface
2. **Frame Processing:** Each frame is processed sequentially:
   - **Person Detection:** YOLO model identifies people in the frame
   - **Object Tracking:** ByteTrack maintains person identities across frames
   - **Group Detection:** DBSCAN clustering algorithm identifies groups (3+ people within 75-pixel radius)
   - **Dwell Time Calculation:** Tracks how long groups persist
3. **Event Logging:** Key events are recorded to CSV:
   - Group formation
   - Group dispersal  
   - Dwell time measurements
4. **Visual Output:** Annotated frames with bounding boxes and group labels
5. **Results Export:** CSV logs and snapshot images available for download

### Critical Functions:
- `VideoProcessor.process_video_and_yield_frames()`: Main processing loop
- `VideoProcessor.process_single_frame()`: Core frame analysis logic
- `VideoProcessor._log_group_event()`: Event logging system

## 5. Dependencies and Integrations

### Core Dependencies:
- **`streamlit`**: Web framework for the user interface and real-time visualization
- **`opencv-python`**: Computer vision library for video processing and image manipulation
- **`ultralytics`**: Provides YOLOv8 object detection model implementation
- **`supervision`**: Computer vision utilities for tracking, annotation, and detection handling
- **`numpy`**: Numerical computing for array operations and mathematical calculations
- **`scikit-learn`**: Machine learning library specifically used for DBSCAN clustering algorithm

### AI Model Integration:
- **YOLOv8n**: Pre-trained object detection model for person detection
- **ByteTrack**: Multi-object tracking algorithm for maintaining person identities
- **DBSCAN**: Density-based clustering for group identification

## 6. Getting Started

Based on the project structure, here are the inferred setup steps:

### Prerequisites:
- Python 3.7+ installed
- Sufficient disk space for video files and output

### Installation:
1. **Clone/Download** the repository to your local machine
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify Model:** Ensure `yolov8n.pt` is in the root directory (should be included)

### Running the Application:
1. **Start the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
2. **Access Interface:** Open your browser to `http://localhost:8501`
3. **Upload Video:** Use the file uploader to select a video file (MP4, MOV, AVI)
4. **Process Video:** Click "Analyze Video" to begin processing
5. **View Results:** Download CSV logs and view snapshot images

### Output Location:
- Results are saved in the `output/` directory
- CSV logs: `output/group_analysis_log.csv`
- Snapshot images: `output/saved_frames/`

## 7. Potential Areas of Interest and Caution

### Areas to Focus On First:
1. **Group Detection Algorithm** (`src/video_processor.py`, lines 95-108): The DBSCAN clustering logic with hardcoded parameters (`eps=75`, `min_samples=3`) that may need tuning based on video resolution and use case.

2. **Performance Optimization** (`src/video_processor.py`, line 108): Frame saving occurs every 30 frames to optimize performance - this could be configurable.

3. **Memory Management**: Large videos may consume significant memory during processing. Consider implementing batch processing for production use.

### Complex/Critical Code Areas:
1. **Multi-object Tracking Logic**: The integration between YOLO detection and ByteTrack tracking requires careful state management.

2. **Real-time Processing Pipeline**: The generator function `process_video_and_yield_frames()` manages complex state for active groups and logging.

3. **Group Lifecycle Management**: The system for tracking group formation, persistence, and dispersal involves complex state transitions.

### Potential Improvements:
1. **Error Handling**: Limited error handling for video file formats, corrupted files, or processing failures.

2. **Configuration**: Hardcoded parameters (clustering distance, minimum group size) should be configurable.

3. **Testing**: The `test.py` file is empty - comprehensive testing framework needed.

4. **Scalability**: Current implementation processes videos sequentially - consider async processing for multiple videos.

### Security Considerations:
- File upload validation could be enhanced
- Output directory management needs consideration for production deployment
- Consider adding authentication for sensitive video content

## Summary

This codebase represents a well-structured computer vision application with clear separation of concerns, but would benefit from enhanced error handling, configuration management, and comprehensive testing before production deployment.

The system successfully combines modern AI technologies (YOLO, ByteTrack, DBSCAN) with a user-friendly web interface to solve real-world problems in crowd behavior analysis. The modular design makes it relatively easy to understand and extend, though attention should be paid to the areas highlighted above for production readiness. 
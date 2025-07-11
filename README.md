# Group Behavior Analysis System

## Main Purpose

- **People Detection and Tracking**: Using computer vision (YOLO model) to detect and track individuals in video footage
- **Group Analysis**: Identifying when people form groups (using DBSCAN clustering)
- **Dwell Time Analysis**: Measuring how long groups stay together
- **Event Logging**: Recording group formation, dispersal, and duration

## Key Features

### 1. Real-time Processing
- Live video processing with frame-by-frame analysis
- Visual annotations showing people and groups
- Progress tracking during processing

### 2. Group Detection
- Minimum of 3 people to form a group
- Spatial clustering to determine group formation
- Tracking individual members within groups

### 3. Data Collection
- Tracking group members
- Recording dwell times
- Capturing snapshots of key moments
- Generating detailed CSV logs

### 4. User Interface (via Streamlit)
- Web-based interface for video upload
- Real-time processing visualization
- Results display with downloadable logs
- Snapshot gallery of detected groups

## Applications

This system is particularly useful for:
- 🏪 Retail analytics (customer behavior)
- 🌳 Public space monitoring
- 👥 Social interaction studies
- 👪 Crowd behavior analysis
- 🔒 Security and surveillance applications

## Technical Stack
- Python
- YOLO (Object Detection)
- DBSCAN (Clustering)
- OpenCV (Computer Vision)
- Streamlit (Web Interface)
# app.py

import streamlit as st
from src.video_processor import VideoProcessor
import tempfile
import os
import cv2

st.set_page_config(layout="wide", page_title="Group Analysis AI")

st.title("📹 AI Group Detection & Analysis")
st.write("Upload a video to detect and track groups of people, calculate their dwell time, and log the events.")

# Add helpful information about group detection
with st.expander("ℹ️ How Group Detection Works", expanded=False):
    st.markdown("""
    **Group Detection Criteria:**
    - 3 or more people within 75 pixels of each other
    - Groups are highlighted with **green bounding boxes**
    - Snapshots are automatically saved when groups are detected
    - Only group snapshots (with green boxes) are displayed below
    
    **What You'll See:**
    - 🟢 **Green boxes** around detected groups
    - 📸 **Snapshots** of frames containing groups
    - 📊 **CSV log** with detailed group analysis
    - ⏱️ **Dwell time** calculations for each group
    """)

# Define output paths at the start
output_dir = "output"
output_frames_dir = os.path.join(output_dir, "saved_frames")
os.makedirs(output_frames_dir, exist_ok=True)

# --- UI for File Upload ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        source_video_path = tfile.name

    st.video(source_video_path)

    if st.button("Analyze Video"):
        output_log_path = os.path.join(output_dir, "group_analysis_log.csv")

        # --- Initialize and run processor ---
        processor = VideoProcessor(
            log_file=output_log_path,
            saved_frames_dir=output_frames_dir
        )
        
        st.write("Processing video... This may take a few moments.")
        progress_bar = st.progress(0)
        st_frame = st.empty()

        # Process video and display frames in real-time
        group_detection_status = st.empty()
        for annotated_frame, progress in processor.process_video_and_yield_frames(source_video_path):
            progress_bar.progress(progress)
            # Convert color from BGR (OpenCV) to RGB (Streamlit)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Simple status update based on progress
            if progress > 0.1:
                group_detection_status.info("🔍 Processing video and detecting groups...")
            if progress > 0.5:
                group_detection_status.success("🟢 Video processing in progress...")

        progress_bar.progress(1.0)
        st.success("Processing Complete!")

        # --- Display Results ---
        st.header("Results")
        st.write(f"Processed video, log file, and frame snapshots have been saved to the '{output_dir}' directory.")
        
        # Provide download link for the log file
        with open(output_log_path, "rb") as file:
            st.download_button(
                label="Download Analysis Log (CSV)",
                data=file,
                file_name="group_analysis_log.csv",
                mime="text/csv"
            )

        # Display group detection snapshots
        with st.expander("📸 Group Detection Snapshots", expanded=True):
            snapshot_files = sorted(os.listdir(output_frames_dir))
            group_snapshots = [f for f in snapshot_files if '_group_' in f]
            
            if group_snapshots:
                # Display up to 5 group snapshots
                cols = st.columns(3)
                for idx, snapshot in enumerate(group_snapshots[:6]):  # Show up to 6 in 3-column grid
                    with cols[idx % 3]:
                        frame_num = snapshot.split('_')[1]
                        group_id = snapshot.split('_group_')[1].split('.')[0]
                        st.image(
                            os.path.join(output_frames_dir, snapshot), 
                            caption=f"Frame {frame_num} - Group {group_id}",
                            use_container_width=True
                        )
                st.info(f"📊 Found {len(group_snapshots)} group detection snapshots across {len(set([f.split('_group_')[1].split('.')[0] for f in group_snapshots]))} unique groups")
            else:
                st.write("No group detection snapshots found. Groups will appear when 3+ people are detected within 75 pixels of each other.")

# Display snapshots with filtering options (moved outside the upload block)
if os.path.exists(output_frames_dir) and os.listdir(output_frames_dir):
    with st.expander("📸 Historical Group Detection Snapshots", expanded=False):
        snapshot_files = sorted(os.listdir(output_frames_dir))
        
        # Filter to only show snapshots that contain groups (green boxes)
        group_snapshots = [f for f in snapshot_files if '_group_' in f]
        
        if group_snapshots:
            col1, col2 = st.columns(2)
            
            with col1:
                display_option = st.selectbox(
                    "Display Mode",
                    ["Latest Group Snapshots", "All Group Snapshots", "Specific Group"]
                )
            
            with col2:
                if display_option == "Specific Group":
                    # Extract unique group IDs from filenames
                    group_ids = sorted(set(
                        int(f.split('_group_')[1].split('.')[0]) 
                        for f in group_snapshots 
                    ))
                    selected_group = st.selectbox("Select Group ID", group_ids)
                    snapshots_to_display = [f for f in group_snapshots if f"group_{selected_group}" in f]
                else:
                    snapshots_to_display = group_snapshots[-5:] if display_option == "Latest Group Snapshots" else group_snapshots

            # Display group snapshots in a grid
            cols = st.columns(3)
            for idx, snapshot in enumerate(snapshots_to_display):
                with cols[idx % 3]:
                    # Extract frame number and group ID for better caption
                    frame_num = snapshot.split('_')[1]
                    group_id = snapshot.split('_group_')[1].split('.')[0]
                    st.image(
                        os.path.join(output_frames_dir, snapshot),
                        caption=f"Frame {frame_num} - Group {group_id}",
                        use_container_width=True
                    )
            
            # Show summary statistics
            st.info(f"📊 Found {len(group_snapshots)} group detection snapshots across {len(set([f.split('_group_')[1].split('.')[0] for f in group_snapshots]))} unique groups")
        else:
            st.write("No group detection snapshots found. Groups will appear when 3+ people are detected within 75 pixels of each other.")
else:
    st.write("No snapshots were saved during this run.")
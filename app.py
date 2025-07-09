# app.py

import streamlit as st
from src.video_processor import VideoProcessor
import tempfile
import os
import cv2

st.set_page_config(layout="wide", page_title="Group Analysis AI")

st.title("📹 AI Group Detection & Analysis")
st.write("Upload a video to detect and track groups of people, calculate their dwell time, and log the events.")

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
        for annotated_frame, progress in processor.process_video_and_yield_frames(source_video_path):
            progress_bar.progress(progress)
            # Convert color from BGR (OpenCV) to RGB (Streamlit)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

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

        # Display some of the saved snapshot images
        st.subheader("Saved Snapshots")
        snapshot_files = sorted(os.listdir(output_frames_dir))
        if snapshot_files:
            # Display up to 5 snapshots
            for snapshot in snapshot_files[:5]:
                st.image(os.path.join(output_frames_dir, snapshot), caption=snapshot)
        else:
            st.write("No snapshots were saved during this run.")

# Display snapshots with filtering options (moved outside the upload block)
if os.path.exists(output_frames_dir) and os.listdir(output_frames_dir):
    st.subheader("Saved Snapshots")
    snapshot_files = sorted(os.listdir(output_frames_dir))
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_option = st.selectbox(
            "Display Mode",
            ["Latest 5 Snapshots", "All Snapshots", "Group-specific Snapshots"]
        )
    
    with col2:
        if display_option == "Group-specific Snapshots":
            # Extract unique group IDs from filenames
            group_ids = sorted(set(
                int(f.split('_group_')[1].split('.')[0]) 
                for f in snapshot_files 
                if '_group_' in f
            ))
            selected_group = st.selectbox("Select Group ID", group_ids)
            snapshots_to_display = [f for f in snapshot_files if f"group_{selected_group}" in f]
        else:
            snapshots_to_display = snapshot_files[-5:] if display_option == "Latest 5 Snapshots" else snapshot_files

    # Display snapshots in a grid
    cols = st.columns(3)
    for idx, snapshot in enumerate(snapshots_to_display):
        with cols[idx % 3]:
            st.image(
                os.path.join(output_frames_dir, snapshot),
                caption=f"Frame: {snapshot.split('_')[1]}",
                use_container_width=True  # Updated from use_column_width
            )
else:
    st.write("No snapshots were saved during this run.")
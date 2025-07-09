"""
This code processes video footage to detect, track, and analyze groups of people. 
It uses computer vision and machine learning to:

    1. Detect individual people
    2. Track their movements
    3. Identify when people form groups
    4. Monitor how long these groups stay together (dwell time)
    5. Log all this information for analysis
"""
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from collections import defaultdict
import csv
import cv2
import os

class VideoProcessor:
    def __init__(self, log_file: str, saved_frames_dir: str):
        # --- File Path ---
        self.log_file = log_file
        self.saved_frames_dir = saved_frames_dir
        os.makedirs(self.saved_frames_dir, exist_ok=True)

        # --- Initialization ---
        self.model = YOLO("yolov8n.pt") # person detection
        self.tracker = sv.ByteTrack()   # object tracking

        # --- Annotators ---
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.5)
        self.group_box_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.GREEN)
        self.group_label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.6, text_color=sv.Color.GREEN)

        # --- Group Logging & Tracking ---
        self.log_data = []
        self._initialize_log_file()
        self.active_groups = {}

    def _initialize_log_file(self):
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "group_id", "member_count", "member_ids", "dwell_time_frames", "saved_frame_path"])

    def _log_group_event(self, data):
         with open(self.log_file, mode='a', newline='') as file:
             writer = csv.writer(file)
             writer.writerow([
                 data["frame"], data["group_id"], data["member_count"], data["member_ids"],
                 data["dwell_time_frames"], data["saved_frame_path"]
             ])

    def process_video_and_yield_frames(self, source_path: str):
        """This generator function processes the video and yields annotated frames."""
        cap = cv2.VideoCapture(source_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.process_single_frame(frame, frame_index)
            
            yield annotated_frame, (frame_index / total_frames)
            frame_index += 1
        
        cap.release()
        
        # Final log for any groups still active at the end
        self._log_final_dwell_times(total_frames)

    def process_single_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        # -- person detection ---
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        # -- person tracking ---
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=detections)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        current_groups_in_frame = set()
        # -- group detection (cluster algo: DBSCAN)---
        if len(detections) >= 3:
            points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            dbscan = DBSCAN(eps=75, min_samples=3)
            cluster_labels = dbscan.fit_predict(points)

            grouped_detections = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:
                    grouped_detections[label].append(detections[i])

            for group_id, group_members in grouped_detections.items():
                current_groups_in_frame.add(group_id)
                group_detection = sv.Detections.merge(group_members)

                if group_id not in self.active_groups:
                    self.active_groups[group_id] = {"start_frame": frame_index, "last_seen_frame": frame_index, "members": []}

                # Save frame and log event
                if frame_index % 30 == 0:  # Optimize: log/save only once per second
                    frame_filename = f"frame_{frame_index}_group_{group_id}.jpg"
                    saved_frame_path = os.path.join(self.saved_frames_dir, frame_filename)
                    cv2.imwrite(saved_frame_path, annotated_frame)
                    
                    member_ids = [mem.tracker_id[0] for mem in group_members if mem.tracker_id]
                    member_ids_str = "-".join(map(str, sorted(member_ids)))
                    
                    log_entry = {
                        "frame": frame_index, "group_id": group_id, "member_count": len(group_members),
                        "member_ids": member_ids_str, "dwell_time_frames": 0, "saved_frame_path": saved_frame_path
                    }
                    self._log_group_event(log_entry)
                    self.active_groups[group_id]['last_log'] = log_entry

                annotated_frame = self.group_box_annotator.annotate(scene=annotated_frame, detections=group_detection)
                # Create labels list with same length as detections
                labels = ["Group #{}".format(group_id)] * len(group_detection)
                annotated_frame = self.group_label_annotator.annotate(
                    scene=annotated_frame, 
                    detections=group_detection, 
                    labels=labels
                )

        disappeared_groups = set(self.active_groups.keys()) - current_groups_in_frame
        for group_id in disappeared_groups:
            start_frame = self.active_groups[group_id]["start_frame"]
            dwell_time = frame_index - start_frame
            last_log = self.active_groups[group_id].get('last_log', {})
            
            log_entry = {
                "frame": frame_index, "group_id": group_id, "member_count": last_log.get("member_count", 0),
                "member_ids": last_log.get("member_ids", ""), "dwell_time_frames": dwell_time, "saved_frame_path": "disappeared"
            }
            self._log_group_event(log_entry)
            del self.active_groups[group_id]

        return annotated_frame
    
    def _log_final_dwell_times(self, total_frames):
        for group_id, group_data in self.active_groups.items():
            start_frame = group_data["start_frame"]
            dwell_time = total_frames - start_frame
            last_log = group_data.get('last_log', {})
            log_entry = {
                "frame": total_frames, "group_id": group_id, "member_count": last_log.get("member_count", 0),
                "member_ids": last_log.get("member_ids", ""), "dwell_time_frames": dwell_time, "saved_frame_path": "video_end"
            }
            self._log_group_event(log_entry)
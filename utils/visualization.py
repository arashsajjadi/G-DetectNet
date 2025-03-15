# utils/visualization.py
"""
Visualization module to overlay detected interactions on images or video frames.
"""

import cv2

def overlay_detections(frame, detections, interactions=None):
    """
    Draw bounding boxes and interaction labels on the frame.
    detections: List of dicts with keys "bbox", "label", and "confidence".
    interactions: Optionally, list of strings describing the interaction per detection.
    """
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        label_text = f"{det['label']}:{det['confidence']:.2f}"
        if interactions is not None and idx < len(interactions):
            label_text += f" | {interactions[idx]}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def show_frame(frame, window_name="HOI Detection"):
    """
    Display a frame in a window.
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

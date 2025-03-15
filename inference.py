# inference.py
"""
Real-time inference script for video streams.
Processes video frames using the feature extractor and GNN model, then overlays detections.
"""

import cv2
import torch
from models.feature_extractor import HOIFeatureExtractor
from models.gnn_model import AdvancedHybridHOIGNN  # Changed from HOIGNN to AdvancedHybridHOIGNN
from utils.graph_utils import construct_graph
from utils.visualization import overlay_detections, show_frame


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load trained GNN model (adjust checkpoint as needed)
    model = AdvancedHybridHOIGNN(in_channels=512, hidden_channels=256, out_channels=10, num_layers=3, gnn_type="GraphSAGE").to(device)
    model.load_state_dict(torch.load("checkpoints/hoi_gnn_epoch50.pth", map_location=device))
    model.eval()

    # Initialize feature extractor
    feature_extractor = HOIFeatureExtractor(device=device)

    # Open video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: Feature Extraction
        output = feature_extractor.process_frame(frame)
        detections = output["detections"]
        features = output["features"]
        keypoints = output["keypoints"]

        # Stage 2: Graph Construction (using distance-based method)
        node_features, edge_index = construct_graph(detections, keypoints, features, threshold=50)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        # Stage 3 & 4: GNN Processing and Interaction Classification
        with torch.no_grad():
            interaction_logits = model(node_features, edge_index)
            # For demonstration, classify interactions per node
            interactions = interaction_logits.argmax(dim=1).cpu().numpy().tolist()
            interaction_names = [f"Interaction {i}" for i in interactions]

        # Stage 5: Visualization
        vis_frame = overlay_detections(frame, detections, interactions=interaction_names)
        show_frame(vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

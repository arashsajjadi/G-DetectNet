# utils/graph_utils.py
"""
Graph construction utilities.
Creates a graph representation (node feature tensor and edge list) from detection outputs.
Improved to compute edges based on Euclidean distance between bounding box centroids.
"""
import torch
import numpy as np


def compute_centroid(bbox):
    """
    Compute centroid (x, y) from a bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array([cx, cy])


def euclidean_distance(pt1, pt2):
    """
    Compute Euclidean distance between two points.
    """
    return np.linalg.norm(pt1 - pt2)


def construct_graph(detections, keypoints, features, threshold=50):
    """
    Construct a graph:
      - Nodes: Each detection (human/object) with its feature vector.
      - Edges: Connect nodes whose bounding box centroids are within a given threshold.
    Returns:
      - x: Node feature tensor [num_nodes, feature_dim]
      - edge_index: Tensor in COO format with shape [2, num_edges]
    """
    num_nodes = len(detections)
    # Stack features (each feature is assumed to be [1, feature_dim])
    x = torch.cat(features, dim=0)  # [num_nodes, feature_dim]

    # Compute centroids for each detection
    centroids = [compute_centroid(det["bbox"]) for det in detections]

    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = euclidean_distance(centroids[i], centroids[j])
            if dist < threshold:
                edges.append([i, j])
                edges.append([j, i])
    if edges:
        edge_index = torch.tensor(edges).t().contiguous()
    else:
        # In case no edges were found, return an empty tensor with appropriate shape
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return x, edge_index

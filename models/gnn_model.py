# gnn_model.py
"""
Enhanced Graph Neural Network module using PyTorch Geometric.
This implementation combines a deep GNN branch with a parallel CNN branch,
and fuses their outputs using multi-head self-attention followed by a linear projection.
The advanced techniques employed include:
- A deeper GNN branch (6 layers) with residual connections, dropout, and layer normalization
  to mitigate over-smoothing and gradient vanishing.
- A parallel CNN branch using multiple 1D convolutional layers to capture local node interactions.
- A fusion module that uses multi-head self-attention to blend the global graph features
  and local CNN features effectively.
- The overall architecture increases the model capacity significantly (checkpoint sizes should be 10× larger).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class AdvancedHybridHOIGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, gnn_type="GraphSAGE", num_heads=4):
        """
        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Hidden layer dimension.
            out_channels (int): Number of output classes.
            num_layers (int): Total number of layers (including first and final).
            gnn_type (str): "GraphSAGE" or "GAT".
            num_heads (int): Number of attention heads (if using GAT or for the fusion attention).
        """
        super(AdvancedHybridHOIGNN, self).__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        # ---------------------------
        # GNN Branch (Deep and Advanced)
        # ---------------------------
        self.gnn_convs = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        # First layer: input -> hidden
        if gnn_type == "GraphSAGE":
            self.gnn_convs.append(SAGEConv(in_channels, hidden_channels))
        else:
            self.gnn_convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.gnn_norms.append(nn.LayerNorm(hidden_channels))
        # Hidden layers (num_layers-2 layers)
        for i in range(1, num_layers - 1):
            if gnn_type == "GraphSAGE":
                self.gnn_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.gnn_convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.gnn_norms.append(nn.LayerNorm(hidden_channels))
        # Final layer (output hidden features for fusion)
        if gnn_type == "GraphSAGE":
            self.gnn_convs.append(SAGEConv(hidden_channels, hidden_channels))
        else:
            self.gnn_convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
        self.gnn_dropout = nn.Dropout(p=0.5)

        # ---------------------------
        # CNN Branch (Processing Node Features as a Sequence)
        # ---------------------------
        # Reshape node features to [batch=1, channels=hidden_channels, length=num_nodes]
        self.cnn_conv1 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.cnn_norm1 = nn.LayerNorm(hidden_channels)
        self.cnn_conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.cnn_norm2 = nn.LayerNorm(hidden_channels)
        self.cnn_conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.cnn_norm3 = nn.LayerNorm(hidden_channels)
        self.cnn_dropout = nn.Dropout(p=0.5)

        # ---------------------------
        # Fusion and Self-Attention
        # ---------------------------
        # After the GNN and CNN branches, fuse features by concatenation.
        self.fusion_linear = nn.Linear(2 * hidden_channels, out_channels)
        # Multi-head self-attention on the fused features.
        self.attention = nn.MultiheadAttention(embed_dim=2 * hidden_channels, num_heads=4, batch_first=True)
        self.attn_dropout = nn.Dropout(p=0.3)
        self.attn_norm = nn.LayerNorm(2 * hidden_channels)

    def forward(self, x, edge_index):
        # ----- GNN Branch -----
        gnn_out = x
        for i, conv in enumerate(self.gnn_convs):
            residual = gnn_out
            gnn_out = conv(gnn_out, edge_index)
            if i < len(self.gnn_norms):
                gnn_out = self.gnn_norms[i](gnn_out)
            gnn_out = F.relu(gnn_out)
            gnn_out = self.gnn_dropout(gnn_out)
            if residual.shape == gnn_out.shape:
                gnn_out = gnn_out + residual  # Residual connection

        # ----- CNN Branch -----
        # Reshape to [1, hidden_channels, num_nodes] for Conv1d
        cnn_input = gnn_out.transpose(0, 1).unsqueeze(0)
        cnn_out = F.relu(self.cnn_conv1(cnn_input))
        cnn_out = self.cnn_norm1(cnn_out.transpose(1, 2)).transpose(1, 2)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = F.relu(self.cnn_conv2(cnn_out))
        cnn_out = self.cnn_norm2(cnn_out.transpose(1, 2)).transpose(1, 2)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = F.relu(self.cnn_conv3(cnn_out))
        cnn_out = self.cnn_norm3(cnn_out.transpose(1, 2)).transpose(1, 2)
        cnn_out = self.cnn_dropout(cnn_out)
        # Reshape back to [num_nodes, hidden_channels]
        cnn_out = cnn_out.squeeze(0).transpose(0, 1)

        # ----- Fusion -----
        fused = torch.cat([gnn_out, cnn_out], dim=1)  # shape: [num_nodes, 2*hidden_channels]
        # Prepare for self-attention: add batch dimension -> [1, num_nodes, 2*hidden_channels]
        fused_seq = fused.unsqueeze(0)
        attn_output, _ = self.attention(fused_seq, fused_seq, fused_seq)
        attn_output = self.attn_dropout(attn_output)
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output.squeeze(0)  # [num_nodes, 2*hidden_channels]
        logits = self.fusion_linear(attn_output)  # [num_nodes, out_channels]
        return logits

if __name__ == "__main__":
    # Dummy test with random data.
    import torch
    num_nodes = 50
    x = torch.randn((num_nodes, 512))
    # Create a simple chain graph.
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes - 1)]).t().contiguous()
    model = AdvancedHybridHOIGNN(in_channels=512, hidden_channels=256, out_channels=10, num_layers=6, gnn_type="GraphSAGE", num_heads=4)
    out = model(x, edge_index)
    print("Output shape:", out.shape)

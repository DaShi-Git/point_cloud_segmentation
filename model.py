import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        #torch.cuda.manual_seed(12345)
        self.conv1 = GCNConv(3, 6)
        self.conv2 = GCNConv(6, 12)
        self.conv3 = GCNConv(12, 24)
        self.classifier = Linear(24, 13)

    def forward(self, x, edge_index):
        # Forward pass 
        h = self.conv1(x, edge_index)
        h = h.tanh()
        #h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.normalize(h, p=2, dim=1)
        h = h.tanh()
        h = self.conv3(h, edge_index)# Final GNN embedding space.
        h = F.normalize(h, p=2, dim=1)
        h = self.classifier(h)
        h = F.normalize(h, p=2, dim=1)
        return h
    
    

import torch

# Create a sample tensor (e.g., predicted logits/probabilities for 2 samples, 4 classes)
logits = torch.tensor([[1.2, 0.5, 5.8, 2.1],
                       [4.3, 8.9, 0.1, 3.2]])

# Get the top 2 largest values and their indices across the last dimension (dim=1)
values, indices = torch.topk(logits, k=2, dim=1)

print("Top Values:\n", values)
print("\nTop Indices:\n", indices)
# results
# Top Values:
#  tensor([[5.8000, 2.1000],
#         [8.9000, 4.3000]])

# Top Indices:
#  tensor([[2, 3],
#         [1, 0]])
def gather_edges(edges, neighbor_idx):
    # edges [B,N,N,C] neighbor_idx [B,N,K]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1)) # [B,N,K,1] -> [B,N,K,C]
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

import torch

# 模拟一个小批次：Batch=1, 节点数N=4, 特征维度C=2, 邻居数K=2
B, N, C, K = 1, 4, 2, 2

# 1. 构造假的 edges: [1, 4, 4, 2]
# 为了好辨认，让 edges[0, i, j, 0] = i*10 + j (代表节点i到j的某种特征)
edges = torch.zeros(B, N, N, C)
for i in range(N):
    for j in range(N):
        edges[0, i, j, 0] = i * 10 + j
        edges[0, i, j, 1] = 99 # 另一个特征通道
print("原始edges:" , edges)

# 2. 构造假的 neighbor_idx: [1, 4, 2]
# 假设节点0的邻居是2和3; 节点1的邻居是0和2 ...
neighbor_idx = torch.tensor([[[2, 3], 
                              [0, 2], 
                              [1, 3], 
                              [0, 1]]]) # shape: [1, 4, 2]

# 3. 执行目标代码
neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
edge_features = torch.gather(edges, 2, neighbors)

print("edge_features shape:", edge_features.shape) # 应该是 [1, 4, 2, 2]
print("edge_features:\n", edge_features)
# 检查 edge_features[0, 0, 0, 0] 是否等于 0*10 + 2 = 2 (节点0的第0个邻居是2)

# Top Indices:
#  tensor([[2, 3],
#         [1, 0]])
# 原始edges: tensor([[[[ 0., 99.],
#           [ 1., 99.],
#           [ 2., 99.],
#           [ 3., 99.]],

#          [[10., 99.],
#           [11., 99.],
#           [12., 99.],
#           [13., 99.]],

#          [[20., 99.],
#           [21., 99.],
#           [22., 99.],
#           [23., 99.]],

#          [[30., 99.],
#           [31., 99.],
#           [32., 99.],
#           [33., 99.]]]])
# edge_features shape: torch.Size([1, 4, 2, 2])
# edge_features:
#  tensor([[[[ 2., 99.],
#           [ 3., 99.]],

#          [[10., 99.],
#           [12., 99.]],

#          [[21., 99.],
#           [23., 99.]],

#          [[30., 99.],
#           [31., 99.]]]])

# ProeteinFeatures 里的几个函数 

# D_neighbors, E_idx = self._dist(Ca, mask)  # （这个E_idx用在了后面的rbf的计算）


#
def gather_nodes(nodes, neighbor_idx):
  # nodes [B, N,C] # neighbor_idx  [B,N, K]
  neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))  # [B, N*K]
  neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2)) # [B, N*K, 1] -> [B, N*K, C]
  # Gather and re-pack
  neighbor_features = torch.gather(nodes, 1, neighbors_flat)  # [B, N+ N*K, C]
  neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
  return neighbor_features
  

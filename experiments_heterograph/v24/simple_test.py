import torch
from GAT import SimpleGAT

print("Testing v24 GAT architecture...")

# Create model
model = SimpleGAT(hidden_dim=16, num_layers=1, dropout=0.0, heads=1)
print("Model created successfully!")

# Create test data
device = torch.device("cpu")  # Use CPU for simplicity
batch_size = 2

x_dict = {
    'user': torch.randn(4, 34),
    'media_session': torch.randn(2, 778),
    'comment': torch.randn(6, 770)
}

edge_index_dict = {
    ('user', 'publishes', 'media_session'): torch.tensor([[0, 1], [0, 1]]),
    ('user', 'creates', 'comment'): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    ('comment', 'belongs_to', 'media_session'): torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
    ('comment', 'mentions', 'user'): torch.tensor([[0, 1], [1, 2]]),
    ('user', 'offensive_comment', 'user'): torch.tensor([[0], [1]]),
    ('user', 'non_offensive_comment', 'user'): torch.tensor([[2], [3]]),
    'batch_dict': {
        'user': torch.tensor([0, 0, 1, 1]),
        'media_session': torch.tensor([0, 1]),
        'comment': torch.tensor([0, 0, 0, 1, 1, 1])
    }
}

print("Input shapes:")
for node_type, features in x_dict.items():
    print(f"  {node_type}: {features.shape}")

# Forward pass
model.eval()
with torch.no_grad():
    try:
        outputs = model(x_dict, edge_index_dict)
        print(f"Output shape: {outputs.shape}")
        print(f"Output: {outputs}")
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

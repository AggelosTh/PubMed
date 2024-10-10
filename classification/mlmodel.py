import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
sbert.max_seq_length = 512

class SBertModel(nn.Module):
    def __init__(self, input_dimension, num_of_labels):
        super(SBertModel, self).__init__()

        self.fc1 = nn.Linear(input_dimension, num_of_labels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x

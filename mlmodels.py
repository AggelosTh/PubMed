import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
sbert.max_seq_length = 256

class SBertModel(nn.Module):
    def __init__(self, input_dimension, num_of_labels):
        super(SBertModel, self).__init__()

        self.fc1 = nn.Linear(input_dimension, 5000)
        self.fc2 = nn.Linear(5000, 2000)
        self.fc3 = nn.Linear(2000, 800)
        self.fc4 = nn.Linear(800, 100)
        self.fc5 = nn.Linear(100, num_of_labels)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.gelu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

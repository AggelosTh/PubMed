import torch.nn as nn
import torch
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
# model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', device=device)

class SBertModel(nn.Module):
    def __init__(self, input_dimension, num_of_labels):
        super(SBertModel, self).__init__()
        # self.dropout5 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.2)
        # self.dropout1 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(input_dimension, 5000)
        # self.bn1 = nn.BatchNorm1d(5000)

        self.fc2 = nn.Linear(5000, 2000)
        # self.bn2 = nn.BatchNorm1d(2000)

        self.fc3 = nn.Linear(2000, 800)
        # self.bn3 = nn.BatchNorm1d(800)

        self.fc4 = nn.Linear(800, 100)
        # self.bn4 = nn.BatchNorm1d(100)

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

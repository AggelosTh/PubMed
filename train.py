from load_data import df
from sklearn.model_selection import train_test_split
from mlmodels import SBertModel, device, sbert
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

X = df['text'].tolist()
y = df['labels'].tolist()

num_of_labels = 14

X_train_text, X_temp_text, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_dev_text, X_test_text, y_dev, y_test = train_test_split(X_temp_text, y_temp, test_size=0.5, random_state=42)

print("Encoding the texts ...")
x_train = sbert.encode(
    [sentence for sentence in X_train_text]
)
x_dev = sbert.encode(
    [sentence for sentence in X_dev_text]
)
x_test = sbert.encode(
    [sentence for sentence in X_test_text]
)

input_dimension = x_train.shape[1]
model = SBertModel(input_dimension, num_of_labels)

print("Convert the data to tensors")
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
x_dev_tensor = torch.tensor(x_dev, dtype=torch.float32).to(device)
y_dev_tensor = torch.tensor(y_dev, dtype=torch.long).to(device)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(x_dev_tensor, y_dev_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = SBertModel(input_dimension, num_of_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_acc = 0
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    
    for inputs, labels in train_loader_tqdm:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        train_loader_tqdm.set_postfix(training_loss=running_loss / len(train_loader))
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    
    val_loss = 0
    model.eval()
    correct = 0
    total = 0

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_acc}')
    
    val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    
    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loader_tqdm.set_postfix(accuracy=correct / total)
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    scheduler.step(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}, Val Accuracy: {val_acc}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = model.state_dict().copy()
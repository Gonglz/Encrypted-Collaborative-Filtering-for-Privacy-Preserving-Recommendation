import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import hashlib
import matplotlib.pyplot as plt

# Manually implement AES encryption and decryption
def aes_encrypt_manual(data, key):
    key = hashlib.sha256(key.encode()).digest()[:16]  # Ensure the key is 16 bytes
    data = data.encode()
    pad_len = 16 - len(data) % 16  # PKCS#7 padding
    padded_data = data + bytes([pad_len] * pad_len)
    encrypted = bytes([(padded_data[i] ^ key[i % len(key)]) for i in range(len(padded_data))])
    return encrypted

def aes_decrypt_manual(encrypted_data, key):
    key = hashlib.sha256(key.encode()).digest()[:16]  # Ensure the key is 16 bytes
    decrypted = bytes([(encrypted_data[i] ^ key[i % len(key)]) for i in range(len(encrypted_data))])
    pad_len = decrypted[-1]  # Last byte indicates the padding length
    return decrypted[:-pad_len].decode()

# AES key
aes_key = "manual_aes_key"

# Load and preprocess data
ratings = pd.read_csv('Dataset/ml-latest-small/ratings.csv')
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
ratings['user_id'] = user_encoder.fit_transform(ratings['userId'])
ratings['movie_id'] = item_encoder.fit_transform(ratings['movieId'])

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

# Split data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=39)
train_users = train_data['user_id'].astype(str).values
train_items = train_data['movie_id'].astype(str).values
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32).unsqueeze(1) / 5.0

test_users = torch.tensor(test_data['user_id'].values, dtype=torch.long)
test_items = torch.tensor(test_data['movie_id'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32).unsqueeze(1) / 5.0

# Encrypt user and item IDs
encrypted_train_users = [aes_encrypt_manual(str(uid), aes_key) for uid in train_users]
encrypted_train_items = [aes_encrypt_manual(str(mid), aes_key) for mid in train_items]

# Define decryption function
def decrypt_ids(encrypted_users, encrypted_items, aes_key):
    decrypted_users = [int(aes_decrypt_manual(data, aes_key)) for data in encrypted_users]
    decrypted_items = [int(aes_decrypt_manual(data, aes_key)) for data in encrypted_items]
    return torch.tensor(decrypted_users, dtype=torch.long), torch.tensor(decrypted_items, dtype=torch.long)

# Model definition
class EnhancedMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=32):
        super(EnhancedMatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.activation_out = nn.Sigmoid()

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        return self.activation_out(self.fc3(x)) * 5.0

# Initialize model
model = EnhancedMatrixFactorization(num_users, num_items)
optimizer = optim.Adam(model.parameters(), lr=0.00005)
loss_fn = nn.MSELoss()

# Decrypt user and item IDs for training
dec_train_users, dec_train_items = decrypt_ids(encrypted_train_users, encrypted_train_items, aes_key)

# Training data loader
train_dataset = TensorDataset(dec_train_users, dec_train_items, train_ratings)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Metrics storage
train_losses = []
test_losses = []
test_rmses = []
test_maes = []

# Training process
epochs = 30
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for users, items, ratings in train_loader:
        predictions = model(users, items)
        loss = loss_fn(predictions, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Testing process
    model.eval()
    with torch.no_grad():
        predictions = model(test_users, test_items)
        test_loss = loss_fn(predictions, test_ratings).item()
        rmse = torch.sqrt(loss_fn(predictions, test_ratings))
        mae = torch.mean(torch.abs(predictions - test_ratings))

        test_losses.append(test_loss)
        test_rmses.append(rmse.item())
        test_maes.append(mae.item())

    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Test Loss={test_loss:.4f}, RMSE={rmse.item():.4f}, MAE={mae.item():.4f}")

# Visualize performance metrics
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
plt.plot(range(1, epochs + 1), test_rmses, label='Test RMSE')
plt.plot(range(1, epochs + 1), test_maes, label='Test MAE')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Performance Metrics')
plt.legend()
plt.grid()
plt.show()

# Sample predictions
print("\nSample Predictions:")
print(f"Actual: {test_ratings[:5].numpy().flatten()}, Predicted: {predictions[:5].numpy().flatten()}")

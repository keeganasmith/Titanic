import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),  
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, 512),
            nn.ReLU(),  
            nn.Linear(512, 512),  
            nn.ReLU(),  
            nn.Linear(512, 512),  
            nn.ReLU(),  
            nn.Linear(512, num_classes)  # Output layer, no activation (logits)
        )

    def forward(self, x):
        return self.layers(x)
    
def prepare_data(df):
    df.dropna(inplace = True)
    df.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"], inplace=True)
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])
    
def train_nn(df):
    df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X = df.drop(columns=["Survived"]).values
    Y = df["Survived"].values
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = NeuralNetwork(input_size=len(df.columns) - 1, num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_features, batch_labels in dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    X_test = test_df.drop(columns=["Survived"]).values
    Y_test = test_df["Survived"].values
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    
    # Evaluate the model on the test set
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(x_test_tensor)  # Get raw predictions (logits)
        _, predicted = torch.max(outputs, 1)  # Get predicted classes
        correct = (predicted == y_test_tensor).sum().item()  # Count correct predictions
        total = y_test_tensor.size(0)  # Total number of samples
        accuracy = correct / total  # Compute accuracy
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
def main():
    df = pd.read_csv("./data/train.csv")   
    print(df["Survived"].value_counts(normalize=True))

    prepare_data(df)
    train_nn(df)
if __name__ == "__main__":
    main()
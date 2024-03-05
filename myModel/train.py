from models import *
from torch.utils.data import DataLoader, random_split
import torch
import time
import datetime
import os
import numpy as np

video_path = 'data/video_frames.mat'
metadata_path = "data/metadata.txt"

def main():
    # Initialize model, loss function and optimizer
    model = VideoTransformer(
        frame_embed_dim=256,
        meta_embed_dim=10,
        n_frames=5,
        n_metadata=10,
        n_classes=4,
        nhead=2,
        num_layers=2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader
    #transform = Compose([Resize((224, 224)), ToTensor()])
    dataset = VideoDataset(video_path, metadata_path)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Training model
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=5):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        for frames, metadata, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(frames, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate model at the end of each epoch
        accuracy, avg_loss = evaluate_model(model, val_dataloader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Save the checkpoint if the performance is better than all previous epochs
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_filename = os.path.join("checkpoints", f"checkpoint_{accuracy:.4f}_{avg_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved as {checkpoint_filename}")

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Switch the model to eval mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # No calculating the gradient during evaluation
        for frames, metadata, labels in dataloader:
            outputs = model(frames, metadata)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Switch the model to train mode
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return accuracy, avg_loss


if __name__ == '__main__':
    main()
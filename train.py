import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import create_dataloaders
from model import create_couple_model
from tqdm import tqdm

def train_model(data_dir, batch_size=32, epochs=20, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_loader, val_loader, class_names = create_dataloaders(data_dir, batch_size=batch_size)
    
    # Create Model
    model = create_couple_model(num_classes=len(class_names))
    model.to(device)
    
    # Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    best_val_acc = 0.0
    early_stop_patience = 5
    early_stop_counter = 0
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_loop.set_postfix(val_loss=loss.item(), val_acc=100.*val_correct/val_total)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%")
        
        # Step the scheduler
        scheduler.step(val_epoch_acc)
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"Saving best model with Acc: {best_val_acc:.2f}%")
            early_stop_counter = 0 # Reset counter
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {early_stop_patience}")
            
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered! Training stopped.")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        train_model(args.data_dir, args.batch_size, args.epochs, args.lr)
    else:
        print(f"Directory {args.data_dir} not found. Please run download_data.py first.")

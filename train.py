import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import CIFAR10_CNN

def train_model():
    # 1. Hardware Selection: We are deploying the RTX 3050 GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DONANIM] Training device: {device}")

    # 2. Data Loaders and Model Initialization
    train_loader, test_loader = get_loaders(batch_size=64)
    model = CIFAR10_CNN().to(device)

    # 3. Loss Function and Optimization Algorithm
    # CrossEntropyLoss: Penalizes classification errors logarithmically (contains Softmax internally).
    criterion = nn.CrossEntropyLoss()
    # Adam: A modern gradient descent algorithm that dynamically adjusts the learning rate on a per-parameter basis.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 10  # The model will scan the entire dataset from start to finish 10 times.
    print("\n--- Egitim Basliyor ---")

    for epoch in range(epochs):
        model.train() # Set the model to training mode (Dropout and BatchNorm active)
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Transfer data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # A) Reset Gradients: Clear derivative accumulations remaining from the previous step
            optimizer.zero_grad()

            # B) Forward Pass: Generate predictions
            outputs = model(inputs)

            # C) Loss Calculation: Measure the difference between true labels and predictions
            loss = criterion(outputs, targets)

            # D) Backpropagation: Calculate gradients using the chain rule
            loss.backward()

            # E) Weight Update: Shift the weights in a direction that reduces the error.
            optimizer.step()

            # Gather statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the success and error rates at the end of each epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss (Hata): {epoch_loss:.4f} | Accuracy (Basari): %{epoch_acc:.2f}")

    # 5. Save Trained Model Weights
    torch.save(model.state_dict(), "cifar10_cnn_weights.pth")
    print("\n[BILGI] Model agirliklari 'cifar10_cnn_weights.pth' olarak basariyla kaydedildi.")

if __name__ == "__main__":
    train_model()
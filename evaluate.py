import torch
from dataset import get_loaders
from model import CIFAR10_CNN

def evaluate_model():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Evaluation Device: {device}")

    # 2. Load Test Data and Initialized Model
    _, test_loader = get_loaders(batch_size=64)
    
    model = CIFAR10_CNN().to(device)
    # Load trained model weights
    model.load_state_dict(torch.load("cifar10_cnn_weights.pth", map_location=device))
    model.eval()  # Set model to evaluation mode (Freezes Dropout & BatchNorm layers)

    # CIFAR-10 Class Labels
    classes = [
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ]
    
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    total_correct = 0
    total_samples = 0

    print("\n--- Evaluating Test Dataset ---")
    
    # Disable gradient computation for memory efficiency and faster inference
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Overall batch accuracy tracking
            total_samples += labels.size(0)
            total_correct += (predictions == labels).sum().item()

            # Class-level accuracy tracking
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # 3. Compute and Display Metrics
    overall_acc = 100 * total_correct / total_samples
    print(f"\n==========================================")
    print(f" Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"==========================================\n")

    print("--- Class-wise Accuracy Breakdown ---")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f" {classname:<12} : {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
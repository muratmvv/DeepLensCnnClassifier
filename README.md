# PyTorch CIFAR-10 Image Classifier (CNN)

A custom Convolutional Neural Network (CNN) architecture implemented in PyTorch for multi-class image classification on the CIFAR-10 dataset.

## Architecture Highlights
- **Convolutional Layers:** 2D Convolutions with 3x3 kernels and Batch Normalization for stable gradients.
- **Regularization:** Spatial Max Pooling (2x2) and Dropout (30%) to prevent overfitting.
- **Optimization:** Cross-Entropy Loss with Adam optimizer ($lr=0.001$).

## Performance & Metrics
- **Overall Test Accuracy:** ~75.56% (10 Epochs)
- **Top Performing Classes:** Truck (86.0%), Ship (85.5%), Automobile (82.9%)

| Class | Accuracy (%) |
|---|---|
| Airplane | 77.50% |
| Automobile | 82.90% |
| Bird | 68.80% |
| Cat | 50.30% |
| Deer | 70.00% |
| Dog | 71.90% |
| Frog | 79.80% |
| Horse | 82.90% |
| Ship | 85.50% |
| Truck | 86.00% |

## Project Structure
```text
├── dataset.py      # Automated CIFAR-10 pipeline & transformations
├── model.py        # Custom 2-block CNN architecture
├── train.py        # Hardware-aware training loop (CUDA/CPU)
├── evaluate.py     # Inference & class-wise accuracy evaluation
├── .gitignore      # Excludes datasets (data/) and checkpoints (*.pth)
└── README.md       # Project documentation

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ColorizationCNN

transform = transforms.ToTensor()

dataset = CIFAR10(root='./dataset', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ColorizationCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, _ in loader:
        gray = images.mean(dim=1, keepdim=True)
        target = images[:, 1:, :, :]

        output = model(gray)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model_no_aug.pth")
print("Model WITHOUT augmentation saved")

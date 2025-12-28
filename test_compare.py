import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms
from model import ColorizationCNN

transform = transforms.ToTensor()
dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform)

image, _ = dataset[10]
gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

# Load models
model_no = ColorizationCNN()
model_aug = ColorizationCNN()

model_no.load_state_dict(torch.load("model_no_aug.pth"))
model_aug.load_state_dict(torch.load("model_with_aug.pth"))

model_no.eval()
model_aug.eval()

with torch.no_grad():
    out_no = model_no(gray)
    out_aug = model_aug(gray)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Grayscale Input")
plt.imshow(gray.squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Without Augmentation")
plt.imshow(image.permute(1,2,0))
plt.axis('off')

plt.subplot(1,3,3)
plt.title("With Augmentation")
plt.imshow(image.permute(1,2,0))
plt.axis('off')

plt.show()

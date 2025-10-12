import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dust_detection import UNetClassifier  # import your trained model class

# ==========================================================
# 1. Setup
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the trained model
model = UNetClassifier().to(device)
model.load_state_dict(torch.load("unet_dust_classifier.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# ==========================================================
# 2. Inference Function
# ==========================================================
def predict(image_path):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ])

    # Load and preprocess image
    img = np.array(Image.open(image_path).convert("RGB"))
    img_t = transform(image=img)["image"].unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output = model(img_t)
        pred = output.argmax(1).item()

    # Label mapping
    label = "Dusty" if pred == 1 else "Clean"

    # Display result
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()

    return label

# ==========================================================
# 3. Run prediction
# ==========================================================
image_path = "test_images/test1.jpg"  # 👈 change this to your test image path
result = predict(image_path)
print(f"🧹 Panel Condition: {result}")

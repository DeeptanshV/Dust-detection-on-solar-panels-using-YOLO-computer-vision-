import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================================
# 1. Model Definition (U-Net Encoder + Classification Head)
# ==========================================================
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(UNetEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.blocks.append(self.conv_block(in_channels, feature))
            in_channels = feature

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            x = self.pool(x)
        return x


class UNetClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(UNetClassifier, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        pooled = self.gap(features).view(features.size(0), -1)
        out = self.fc(pooled)
        return out


# ==========================================================
# 2. Dataset Class
# ==========================================================
class SolarDustClassifierDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images, self.labels = self._load_labels(labels_file)

    def _load_labels(self, labels_file):
        images, labels = [], []
        with open(labels_file, 'r') as f:
            for line in f:
                img, lbl = line.strip().split(',')
                images.append(img)
                labels.append(int(lbl))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ==========================================================
# 3. Training Loop
# ==========================================================
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        preds = model(data)
        loss = loss_fn(preds, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (preds.argmax(1) == targets).sum().item()
        total += targets.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


# ==========================================================
# 4. Setup
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2(),
])

# Example labels.txt content:
# panel_1.jpg,0
# panel_2.jpg,1
# 0 = Clean, 1 = Dusty
train_dataset = SolarDustClassifierDataset(
    img_dir="dataset",
    labels_file="dataset/labels.txt",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = UNetClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ==========================================================
# 5. Training
# ==========================================================
epochs = 85
for epoch in range(epochs):
    loss, acc = train_fn(train_loader, model, optimizer, loss_fn, device)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

torch.save(model.state_dict(), "unet_dust_classifier.pth")
print("✅ Model saved as unet_dust_classifier.pth")


# ==========================================================
# 6. Inference
# ==========================================================
def predict(model, image_path):
    model.eval()
    img = np.array(Image.open(image_path).convert("RGB"))
    transform_infer = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2(),
    ])
    img_t = transform_infer(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        pred = output.argmax(1).item()

    label = "Dusty" if pred == 1 else "Clean"
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()
    return label

model.load_state_dict(torch.load("unet_dust_classifier.pth", map_location="cpu"))



def predict(model, image_path):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2(),
    ])

    img = np.array(Image.open(image_path).convert("RGB"))
    img_t = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t).argmax(1).item()

    label = "Dusty" if pred == 1 else "Clean"
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()

    return label

# Example:
predict(model, "test_images/test1.jpg")



# ==========================================================
# 5. Training (with Auto-Save & Interrupt Safety)
# ==========================================================
epochs = 85
start_epoch = 1

try:
    for epoch in range(start_epoch, epochs + 1):
        loss, acc = train_fn(train_loader, model, optimizer, loss_fn, device)
        print(f"Epoch [{epoch}/{epochs}] | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

        # 🔹 Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = f"checkpoints/unet_dust_classifier_epoch{epoch}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

except KeyboardInterrupt:
    # 🔹 Save immediately on Ctrl+C
    print("\n🛑 Training interrupted by user. Saving model before exit...")
    torch.save(model.state_dict(), "final_model_interrupted.pth")
    print("✅ Model saved as final_model_interrupted.pth")

finally:
    # 🔹 Always save the latest weights even if finished normally
    torch.save(model.state_dict(), "unet_dust_classifier_final.pth")
    print("✅ Final model saved safely as unet_dust_classifier_final.pth")


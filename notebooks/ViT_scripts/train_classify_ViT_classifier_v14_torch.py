#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image

# Load your DataFrame
df = pd.read_csv("notebook_results/mmaction_train_data_v14-inclusive-imgs/train_and_test.csv")
out_dir = Path("ViT_resutls")
out_dir.mkdir(exist_ok=True)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["img_path"]
        image = Image.open(img_path)
        label = int(self.dataframe.iloc[idx]["label_index"])

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5)),
    ]
)

# Split your dataset
train_df = df[df["split"] == "train"]
valid_df = df[df["split"] == "test"]

# Debug: randomly select 1000 samples from each dataset
# train_df = train_df.sample(1000)
# valid_df = valid_df.sample(1000)

train_dataset = CustomDataset(train_df, transform=transform)
valid_dataset = CustomDataset(valid_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=32)

# Load and modify the ViT model
model = models.vit_b_16(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device:", device)
model.to(device)


# In[ ]:


# Training Loop
num_epochs = 10

import tqdm

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(train_loader, "training at epoch " + str(epoch + 1), total=len(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(valid_loader, "validation at epoch " + str(epoch + 1), total=len(valid_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Report validation metrics

        print(
            f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(valid_loader)}, Validation Accuracy: {100 * correct / total}%"
        )

    print(
        f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(valid_loader)}, Validation Accuracy: {100 * correct / total}%"
    )

    # Save the model
    torch.save(model.state_dict(), out_dir / (str(epoch + 1) + ".pth"))

ALL CODES
TONGUE PYTHON:
Class.py:
CLASS_ID_TO_MERGED = {
    0: "PALE",
    2: "RED",
    3: "PURPLE",
    6: "SPOTS",
    7: "CRACKED",
    9: "WHITE_COAT",
    10: "YELLOW_COAT",
    11: "BLACK_COAT",
    8: "NORMAL"
}


# =========================================
# Final Merged Class → Index (for training)
# =========================================
# IMPORTANT:
# Old 8 classes remain unchanged.
# NORMAL added as new class with index 8.

MERGED_CLASSES = {
    "PALE": 0,
    "RED": 1,
    "PURPLE": 2,
    "WHITE_COAT": 3,
    "YELLOW_COAT": 4,
    "BLACK_COAT": 5,
    "SPOTS": 6,
    "CRACKED": 7,
    "NORMAL": 8,  # Newly added healthy tongue class
}

Dataset.py:
import os
from PIL import Image
from torch.utils.data import Dataset

from classes import CLASS_ID_TO_MERGED, MERGED_CLASSES


class TongueDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        image_files = set(os.listdir(image_dir))

        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue

            image_name = label_file.replace(".txt", ".jpg")
            if image_name not in image_files:
                continue

            label_path = os.path.join(label_dir, label_file)

            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines:
                continue

            # Take the first object label only (paper-style)
            class_id = int(lines[0].split()[0])

            if class_id not in CLASS_ID_TO_MERGED:
                continue

            merged_name = CLASS_ID_TO_MERGED[class_id]
            merged_id = MERGED_CLASSES[merged_name]

            self.samples.append((image_name, merged_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


Model.py:
import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights
)


def get_model(model_name="mobilenetv2", num_classes=9):

    # ---------------- MobileNetV2 ----------------
    if model_name == "mobilenetv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------- ResNet50 ----------------
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # ---------------- EfficientNet-B0 ----------------
    elif model_name == "efficientnetb0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------- DenseNet121 ----------------
    elif model_name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Invalid model name. Choose from: mobilenetv2, resnet50, efficientnetb0, densenet121")

    return model


Train.py:
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import TongueDataset
from model import get_model


def main():

    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- Transforms ----------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ---------------- Dataset ----------------
    dataset = TongueDataset(
        image_dir="data/images",
        label_dir="data/labels",
        transform=transform
    )

    print("Total samples:", len(dataset))

    # ---------------- Split ----------------
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print("Train samples:", len(train_set))
    print("Validation samples:", len(val_set))
    print("Test samples:", len(test_set))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # ---------------- Models To Train ----------------
    model_names = ["mobilenetv2", "resnet50", "efficientnetb0", "densenet121"]

    epochs = 10
    results = {}

    for model_name in model_names:

        print("\n======================================")
        print(f"Training Model: {model_name}")
        print("======================================")

        model = get_model(model_name=model_name, num_classes=9).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # -------- Training --------
        for epoch in range(epochs):

            model.train()
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total

            # -------- Validation --------
            model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = 100 * correct / total

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # -------- Test Evaluation --------
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")

        print("\nFinal Test Accuracy:", round(test_acc, 2), "%")
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-Score:", round(f1, 4))

        torch.save(model.state_dict(), f"tcm9_{model_name}.pth")
        print(f"Saved model: tcm9_{model_name}.pth")

        results[model_name] = {
            "test_acc": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # ---------------- Final Comparison ----------------
    print("\n\n========= MODEL COMPARISON =========")
    for name, metrics in results.items():
        print(f"\n{name}")
        print("Test Accuracy :", round(metrics["test_acc"], 2))
        print("Precision     :", round(metrics["precision"], 4))
        print("Recall        :", round(metrics["recall"], 4))
        print("F1 Score      :", round(metrics["f1"], 4))


if __name__ == "__main__":
    main()

Predict.py:
import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

from model import get_model
from classes import MERGED_CLASSES


# Reverse class mapping
IDX_TO_CLASS = {v: k for k, v in MERGED_CLASSES.items()}


# =====================================
# FULL TCM + SIDDHA TABLE (INCLUDING NORMAL)
# =====================================

TCM_FULL_INFO = {

    "NORMAL": {
        "tcm_description": "Healthy tongue, balanced Qi",
        "diseases": "No major pathology detected",
        "vatham": "Balanced circulation and nerve function",
        "pitham": "Balanced digestion and metabolism",
        "kabam": "Stable immunity and body fluids",
        "dosha_percent": {"Vatham": 33, "Pitham": 34, "Kabam": 33}
    },

    "PALE": {
        "tcm_description": "Qi and blood deficiency",
        "diseases": "Anemia, Chronic fatigue, Hypothyroidism",
        "vatham": "Fatigue, weakness, cold intolerance, dizziness",
        "pitham": "—",
        "kabam": "Sluggish metabolism, lethargy, low appetite",
        "dosha_percent": {"Vatham": 60, "Pitham": 10, "Kabam": 30}
    },

    "RED": {
        "tcm_description": "Heat syndrome",
        "diseases": "Gastritis, Hyperacidity, Liver disorder",
        "vatham": "—",
        "pitham": "Burning sensation, irritability, loose stools, headache",
        "kabam": "—",
        "dosha_percent": {"Vatham": 10, "Pitham": 80, "Kabam": 10}
    },

    "PURPLE": {
        "tcm_description": "Blood stasis",
        "diseases": "Circulatory disorder, Chronic inflammation",
        "vatham": "Poor circulation, numbness, stagnation",
        "pitham": "Inflammatory congestion, blood heat",
        "kabam": "Fluid retention, heaviness",
        "dosha_percent": {"Vatham": 40, "Pitham": 30, "Kabam": 30}
    },

    "WHITE_COAT": {
        "tcm_description": "Cold syndrome / Dampness",
        "diseases": "Candida, IBS, Weak digestion",
        "vatham": "—",
        "pitham": "—",
        "kabam": "Bloating, mucus formation, heaviness, lethargy",
        "dosha_percent": {"Vatham": 10, "Pitham": 10, "Kabam": 80}
    },

    "YELLOW_COAT": {
        "tcm_description": "Damp-heat",
        "diseases": "Acid reflux, Peptic ulcer, Liver disorder",
        "vatham": "—",
        "pitham": "Hyperacidity, burning chest, skin rashes, anger",
        "kabam": "—",
        "dosha_percent": {"Vatham": 10, "Pitham": 75, "Kabam": 15}
    },

    "BLACK_COAT": {
        "tcm_description": "Severe internal disorder",
        "diseases": "Chronic constipation, Fungal infection",
        "vatham": "Severe dryness, hard stools, irregular bowel",
        "pitham": "Toxic heat (advanced stage), foul breath",
        "kabam": "—",
        "dosha_percent": {"Vatham": 50, "Pitham": 40, "Kabam": 10}
    },

    "SPOTS": {
        "tcm_description": "Blood heat / Toxicity",
        "diseases": "Hormonal imbalance, Migraine",
        "vatham": "Dark spots → poor circulation, anxiety",
        "pitham": "Red spots → headache, inflammation",
        "kabam": "—",
        "dosha_percent": {"Vatham": 40, "Pitham": 50, "Kabam": 10}
    },

    "CRACKED": {
        "tcm_description": "Yin deficiency",
        "diseases": "Anxiety disorder, IBS, Colon dysfunction",
        "vatham": "Constipation, insomnia, nervous tension",
        "pitham": "If red base → dehydration with heat",
        "kabam": "—",
        "dosha_percent": {"Vatham": 70, "Pitham": 20, "Kabam": 10}
    }
}


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Tongue Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    return file_path


def predict(image_path, model_path="tcm9_mobilenetv2.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # 9-class model
    model = get_model(model_name="mobilenetv2", num_classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        pred_idx = torch.argmax(outputs, dim=1).item()

    pred_class = IDX_TO_CLASS[pred_idx]
    info = TCM_FULL_INFO[pred_class]

    # Determine dominant dosha
    dosha_percent = info["dosha_percent"]
    dominant = max(dosha_percent, key=dosha_percent.get)

    # Get only dominant dosha symptoms
    dominant_symptoms = info[dominant.lower()]

    print("\n================ PREDICTION RESULT ================")
    print("Predicted TCM Class :", pred_class)
    print("TCM Description     :", info["tcm_description"])
    print("Associated Diseases :", info["diseases"])

    print("\nDominant Dosha :", dominant)
    print("Dominant Dosha Symptoms :", dominant_symptoms)
    print("====================================================\n")


if __name__ == "__main__":
    img_path = select_image()

    if img_path:
        predict(img_path)
    else:
        print("No image selected.")



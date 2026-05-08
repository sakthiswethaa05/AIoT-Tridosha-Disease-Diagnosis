import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

from .model import get_model
from .classes import MERGED_CLASSES


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


def predict(image_path, model_path="tongue/tcm9_mobilenetv2.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Load model
    model = get_model(model_name="mobilenetv2", num_classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        pred_idx = torch.argmax(outputs, dim=1).item()

    pred_class = IDX_TO_CLASS[pred_idx]
    info = TCM_FULL_INFO[pred_class]

    # Dominant dosha
    dosha_percent = info["dosha_percent"]
    dominant = max(dosha_percent, key=dosha_percent.get)

    dominant_symptoms = info[dominant.lower()]

    # -------- PRINT (keep this) --------
    print("\n================ PREDICTION RESULT ================")
    print("Predicted TCM Class :", pred_class)
    print("TCM Description     :", info["tcm_description"])
    print("Associated Diseases :", info["diseases"])
    print("\nDominant Dosha :", dominant)
    print("Dominant Dosha Symptoms :", dominant_symptoms)
    print("====================================================\n")

    # ✅ -------- RETURN (VERY IMPORTANT) --------
    return {
        "dosha": dominant,
        "class": pred_class,
        "symptoms": dominant_symptoms
    }


if __name__ == "__main__":
    img_path = select_image()

    if img_path:
        result = predict(img_path)
        print(result)
    else:
        print("No image selected.")


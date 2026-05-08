# =========================================
# Original YOLO Class ID → Merged Class Name
# (Paper-style mapping)
# =========================================

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

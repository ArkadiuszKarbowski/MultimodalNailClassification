import re
from pathlib import PurePath


def file_pattern_match(filename):
    """
    File pattern: [PatientID] [LimbCode] [Digit] [Position] [UV].jpg
    Components (space-separated):
    - PatientID: digits
    - LimbCode: SL/SP/RL/RP -> Left foot / Right foot / Left hand / Right hand
    - Digit: 1-5 -> Toe/Finger number
    - Position: H/P -> Horizontal/Parallel orientation
    - Optional: UV -> UV image
    Example: "123 SL 2 H UV.jpg"
    """

    file_pattern = re.compile(r"^(\d+)\s+(SL|SP|RL|RP)\s+([1-5])\s+(H|P)\s*(UV)?\.jpg$")
    return file_pattern.match(filename)


def parse_filename(filename):
    match = file_pattern_match(filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected format.")
    parts = match.groups()
    limb_mapping = {
        "SL": "left_foot",
        "SP": "right_foot",
        "RL": "left_hand",
        "RP": "right_hand",
    }

    info = {
        "patient_id": int(parts[0]),
        "limb": limb_mapping[parts[1]],
        "digit": int(parts[2]),
        "position": parts[3],
        "is_uv": bool(parts[4]),
    }

    return info


def get_img_metadata(image_path):
    filename = PurePath(image_path).name
    class_mapping = {
        "ŁUSZCZYCA": "psoriasis",
        "ONYCHOMYKOZA": "onychomycosis",
        "ONYCHODYSTROFIA MECHANICZNA": "mechanical_dystrophy",
        "ZDROWE PAZNOKCIE": "healthy",
    }

    class_folder = PurePath(image_path).parts[-3]
    info = {"class": class_mapping[class_folder]}
    info.update(parse_filename(filename))
    return info


if __name__ == "__main__":
    image_paths = [
        "data/train/madeup/123 RL 2 P.jpg",
        "data/train/madeup/456 SL 3 H UV.jpg",
        "data/train/madeup/789 SP 1 H.jpg",
        "data/train/madeup/101 RP 5 P UV.jpg",
    ]
    for image_path in image_paths:
        print(get_img_metadata(image_path))

"""Central configuration. Every threshold used by the pipeline lives here (nothing is hard-coded elsewhere)."""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"        # currency classes: dataset/<CODE>_<VALUE>/  (INR_500, TRY_100, USD_20 ...)
NEGATIVE_DIR = BASE_DIR / "negative"      # NOT-currency images: negative/<group>/  (face, phone, wall, ...)
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
# Training checkpoints (resume after a crash / disconnect). On Colab point this at Google Drive, e.g.
#   os.environ["CURRENCY_CKPT_DIR"] = "/content/drive/MyDrive/currency_ckpt"      (before running the training scripts)
CKPT_DIR = Path(os.environ.get("CURRENCY_CKPT_DIR", BASE_DIR / "checkpoints"))

CLASSIFIER_WEIGHTS = MODEL_DIR / "currency_resnet50.weights.h5"
WEIGHTS_PATH = CLASSIFIER_WEIGHTS         # backward-compatible alias
CLASSES_PATH = MODEL_DIR / "class_names.json"
CLASSIFIER_META = MODEL_DIR / "classifier_meta.json"
DETECTOR_WEIGHTS = MODEL_DIR / "currency_detector.weights.h5"
DETECTOR_META = MODEL_DIR / "detector_meta.json"

IGNORED_DATASET_FOLDERS = {"NOT_CURRENCY"}   # legacy folder from older versions - never used as a class
NEGATIVE_GROUPS = ["person", "face", "phone", "laptop", "book", "table", "wall",
                   "clothes", "random_objects", "background", "hard_negatives"]

# Set CURRENCY_NO_PRETRAINED=1 to skip ImageNet weights (offline smoke tests only - never for real training)
PRETRAINED = os.environ.get("CURRENCY_NO_PRETRAINED", "0") != "1"

# ----------------------------------------------------------------------------- shared preprocessing / training
IMG_SIZE = (224, 224)       # identical for training AND inference (see preprocessing.py)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

# Denomination classifier (ResNet50)
EPOCHS_HEAD = 10
EPOCHS_FINE = 15
FINE_TUNE_LAYERS = 30
MAX_IMAGES_PER_CLASS = None   # e.g. 600 to cap very large classes (None = use all)

# Currency / non-currency detector. DETECTOR_BACKBONE: "cnn" = plain CNN trained from scratch, "resnet50" = transfer learning
DETECTOR_BACKBONE = "cnn"
DETECTOR_EPOCHS_HEAD = 6
DETECTOR_EPOCHS_FINE = 6      # (resnet50 only; the plain CNN trains for HEAD + FINE epochs in a single phase)
DETECTOR_FINE_TUNE_LAYERS = 40
DETECTOR_MAX_POSITIVES_PER_CLASS = 400
DETECTOR_COMPOSITES = 1000    # synthetic "note on a background" positives (train split only)
DETECTOR_MIN_NEGATIVES = 300  # warn below this - the detector will be weak

# Data augmentation profiles (applied to TRAINING data only, inside the tf.data pipeline).
# Colour changes are deliberately mild for the denomination classifier: notes are told apart by colour.
CLASSIFIER_AUG = dict(rotation_deg=12, zoom=(0.85, 1.15), shift=0.08, perspective=0.06, flip180_prob=0.25,
                      brightness=(0.75, 1.25), contrast=(0.8, 1.25), saturation=(0.85, 1.15),
                      color_cast=0.06, noise=6.0, lowres_prob=0.35)
DETECTOR_AUG = dict(rotation_deg=25, zoom=(0.7, 1.2), shift=0.15, perspective=0.10, flip180_prob=0.3,
                    brightness=(0.6, 1.4), contrast=(0.7, 1.4), saturation=(0.6, 1.4),
                    color_cast=0.12, noise=10.0, lowres_prob=0.5)

# ----------------------------------------------------------------------------- rejection thresholds
# Stage B - currency / non-currency detector (probability that the ROI is a banknote)
DETECTOR_THRESHOLD = 0.80        # >= this: accepted as currency
DETECTOR_SOFT_THRESHOLD = 0.50   # between soft and threshold: "probably a note, but unusable" -> guidance messages
# Stage C - open-set rejection on the denomination classifier output
CONFIDENCE_THRESHOLD = 0.75      # max softmax probability
MARGIN_THRESHOLD = 0.20          # top-1 minus top-2 probability
ENTROPY_THRESHOLD = 0.60         # normalised entropy (0 = certain, 1 = uniform); must be <= this

# ----------------------------------------------------------------------------- note localisation + quality (OpenCV)
MIN_IMAGE_SIDE = 160             # reject images whose shorter side is smaller
CANDIDATE_MIN_AREA_RATIO = 0.03  # smallest note-like region that is even considered (fraction of frame)
NOTE_MIN_AREA_RATIO = 0.12       # live mode: smaller than this -> "Move the note closer"
NOTE_ASPECT_RANGE = (1.4, 3.2)   # long side / short side of a banknote (most notes are ~2.0-2.5)
RECTANGULARITY_MIN = 0.80        # region area / its bounding rotated-rectangle area
BORDER_MARGIN_RATIO = 0.01       # note touching the frame border within this margin -> "partial"
WARP_LONG_SIDE = 448             # pixels of the long side of the rectified ROI
BLUR_THRESHOLD = 60.0            # variance of Laplacian (ROI normalised to 400 px wide); below -> "too blurry"
BRIGHTNESS_RANGE = (45, 225)     # mean gray level of the ROI
ALLOW_GUIDE_BOX_FALLBACK = False # live: if no note-shaped region is found, classify the guide box anyway (still gated by detector)
GUIDE_WIDTH_FRAC = 0.70          # guide box width as a fraction of frame width
GUIDE_ASPECT = 2.2

# ----------------------------------------------------------------------------- temporal smoothing (webcam)
PROCESS_EVERY_N_FRAMES = 3
SMOOTH_WINDOW = 7                # last N analysed frames kept
SMOOTH_STABLE = 5                # same label needed this many times inside the window
SMOOTH_ABSENT = 4                # this many consecutive misses clear the displayed result

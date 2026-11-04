# 🌍 Currency Recognition — India 🇮🇳 · Turkey 🇹🇷 · USA 🇺🇸

Show a banknote to the webcam (or upload a photo) → **country, denomination, confidence and a one-line description**.
Faces, objects, walls and other non-notes are rejected with **“No currency detected”**.

```
Camera frame / uploaded image
        │
 [A] Note localisation  (OpenCV: edges + contours, aspect ratio, rectangularity, perspective correction)
        │  no note-shaped region ─────────────────────────────► "No currency detected"
 [B] Currency / non-currency detector  (plain CNN trained from scratch, with an explicit negative/ dataset; ResNet50 optional via DETECTOR_BACKBONE)
        │  not a banknote ────────────────────────────────────► "No currency detected"
 Geometry + quality checks  (too far · partial note · blurry · too dark/bright · low resolution)
        │  fails ─────────────────────────────────────────────► guidance message
 [C] Denomination classifier  (ResNet50 transfer learning)
        │  open-set check: max-prob ≥ 0.75, margin ≥ 0.20, normalised entropy ≤ 0.60
        │  fails ─────────────────────────────────────────────► "Unidentified note"
        ▼
 Country + denomination + confidence ──► temporal smoothing (webcam) ──► stable display
```

`webcam.py`, `app.py` and `predict.py` all call the same `CurrencyRecognitionPipeline` (`pipeline.py`).

## Why the old model called faces “currency”
A softmax classifier outputs probabilities that **sum to 1 over its own classes**. It was never shown a non-note, so every
input – a face, a wall – is forced onto the nearest note class, often with high confidence. A threshold cannot fix that;
the model has no concept of “none of the above”. The fix is architectural: localise a note first, then ask a model *trained
on negatives* whether it is a note at all, and only then classify the denomination (plus margin/entropy rejection).
`train.py` prints `outputs/open_set_report.txt` showing how the classifier alone behaves on your negative images.

## Folder layout
```
dataset/<CODE>_<VALUE>/   real notes:   INR_10 … INR_2000, TRY_5 … TRY_200, USD_1 … USD_100
negative/<group>/         NOT notes:    person face phone laptop book table wall clothes random_objects background hard_negatives
model/                    trained files (see below)
outputs/                  reports, confusion matrices, curves
```

## Files
| File | Purpose |
|---|---|
| `config.py` | **all thresholds and paths** |
| `preprocessing.py` | the single preprocessing + augmentation pipeline (train = inference) |
| `dataset_utils.py` | dataset discovery, reproducible stratified split, tf.data |
| `model.py` | ResNet50 denomination classifier |
| `currency_detector.py` | CNN / ResNet50 currency / non-currency model |
| `note_localizer.py` | OpenCV note finder, perspective warp, quality checks |
| `inference.py` | classifier loading + open-set assessment |
| `pipeline.py` | `CurrencyRecognitionPipeline`, `ResultSmoother` |
| `train_detector.py`, `train.py` | training (detector first, then classifier) - **resumable** |
| `checkpointing.py` | saves/restores training state after every epoch (resume after a crash or disconnect) |
| `webcam.py`, `app.py`, `predict.py` | live camera, Streamlit, CLI |
| `prepare_data.py`, `prepare_negatives.py`, `capture_images.py`, `make_folders.py`, `check_dataset.py` | dataset tools |
| `Colab_Training.ipynb` | full training in Google Colab |
| `tests/` | offline unit tests |

## Datasets (Kaggle, downloaded by `prepare_data.py`)
- India: `vishalmane109/indian-currency-note-images-dataset-2020`
- Turkey: `baltacifatih/turkish-lira-banknote-dataset` (5–200 TL)
- USA: `aishwaryatechie/usd-bill-classification-dataset` (manually cropped notes; check that a **$20** folder exists – `check_dataset.py` lists every class. A missing denomination simply won’t be recognised.)

These datasets contain **cropped note images only (no bounding boxes)**, so this project does *not* train a bounding-box object detector.
Localisation is classical computer vision (OpenCV) and the learned detector is a banknote / not-banknote classifier on that crop.
For robust detection of notes in cluttered scenes you would need an annotated detection dataset (e.g. YOLO format) — not included.

## Google Colab (recommended for training)
1. Open `Colab_Training.ipynb` in Colab → Runtime → **T4 GPU**.
2. Run all cells (upload this project zip when asked). Equivalent commands:
```bash
pip install -q kagglehub scikit-learn matplotlib tensorflow-datasets
python prepare_data.py          # currency images from Kaggle
python prepare_negatives.py     # faces/objects/backgrounds from Caltech101  (+ upload your own negatives)
python check_dataset.py
python train_detector.py        # stage B
python train.py                 # stage C
```
3. Download `trained_models.zip` and unzip into your local `model/` folder:
`currency_resnet50.weights.h5`, `class_names.json`, `classifier_meta.json`, `currency_detector.weights.h5`, `detector_meta.json`.

## Resuming interrupted training (checkpoints)
Both `train.py` and `train_detector.py` save their **full training state after every epoch** (weights, optimizer state,
epoch counter, best model, metric log) into `checkpoints/` (`config.CKPT_DIR`). If the runtime dies, the network drops or the
GPU quota runs out, **run the same command again**:

```
[checkpoint] found earlier run in checkpoints/classifier: finished phases=['head'], unfinished={'fine': 7} ...
[checkpoint] phase 'head' already finished - skipping
[checkpoint] phase 'fine': resuming after epoch 7/15
```

- Finished phases (head / fine-tune) are skipped; an unfinished phase continues from its last completed epoch with the same
  optimizer state and learning rate. At most the one epoch that was running when it died is repeated.
- If the dataset, class list or image size changed since the checkpoint was written, the old checkpoint is discarded with a
  warning (it would not match the new data). `--fresh` discards it on purpose: `python train.py --fresh`.
- Re-running after everything finished does not train again; it just re-creates the reports.
- The model files the app loads (`model/*.weights.h5`) are written when training completes.
- **Google Colab:** the local disk is wiped with the runtime, so the notebook (cell 1b) mounts Google Drive and sets
  `CURRENCY_CKPT_DIR=/content/drive/MyDrive/currency_ckpt`. After a disconnect: reconnect, re-run cells 1 - 5 (they rebuild the
  dataset), then the training cell - it resumes from Drive.
- Not restored exactly: the early-stopping patience counter and the ReduceLROnPlateau patience counter restart after a resume.

## Local Windows
```bat
cd /d "D:\NNDL PROJEECTT\currency_recognition"
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

:: collect YOUR negatives (face, room, objects) – the most important step for webcam accuracy
python capture_images.py negative/face
python capture_images.py negative/background
python capture_images.py negative/random_objects

python -m unittest discover tests      :: offline self-check
python webcam.py                       :: live camera (needs trained model/ files)
python -m streamlit run app.py         :: web app
python predict.py note.jpg             :: command line (add --json for JSON)
```
Training locally (slow on CPU): `python prepare_data.py && python train_detector.py && python train.py`.

### Webcam keys
`Q` quit · `D` debug numbers (detector score, margin, entropy, blur, brightness) · `R` reset · `S` save frame ·
**`N` save the current frame/ROI as a hard negative** (use it whenever the app wrongly sees a note, then re-train the detector).

## Tuning (all in `config.py`)
`DETECTOR_THRESHOLD`, `DETECTOR_SOFT_THRESHOLD`, `CONFIDENCE_THRESHOLD`, `MARGIN_THRESHOLD`, `ENTROPY_THRESHOLD`,
`NOTE_MIN_AREA_RATIO`, `BLUR_THRESHOLD`, `BRIGHTNESS_RANGE`, `SMOOTH_*`, `PROCESS_EVERY_N_FRAMES`.
`train_detector.py` prints a suggested detector threshold and the false-accept rate per negative group.
If the OpenCV step misses a note on a low-contrast surface, set `ALLOW_GUIDE_BOX_FALLBACK = True` (the guide box is then
classified, still gated by the detector).

## Result fields
`status` (`ok`, `no_currency`, `unidentified`, `too_blurry`, `too_far`, `partial_note`, `too_dark`, `too_bright`, `low_resolution`),
`is_currency`, `label`, `name`, `description`, `confidence`, `reason`, `detector_score`, `margin`, `entropy`, `top3`.
`confidence` is the confidence in the reported verdict (classifier probability for `ok`/`unidentified`, `1 − detector score` for `no_currency`).

## Limitations (honest notes)
- The detector is only as good as its negatives. With only generic Caltech101 negatives it will still be fooled by things it
  never saw (your room, your phone). Capture your own negatives and use `N` to mine hard negatives.
- Public datasets contain many near-duplicate images, so validation accuracy is optimistic; test with your own photos.
- Only INR, TRY and USD notes are classified. Other currencies are reported as “Unidentified note” **at best** – a closed
  classifier cannot be guaranteed to recognise unseen currencies; the margin/entropy checks reduce, not eliminate, that risk.
- Old and new note series (e.g. older Turkish or Indian issues) are covered only if present in the training data.
- Models trained with the previous version of this project are **not compatible** (architecture changed) – re-train.

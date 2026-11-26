# Executing the RDD2022 YOLO Baseline Training Notebook

This guide explains how to run `RDD2022_YOLO_Baseline_Training.ipynb` in Google Colab, from preparing the dataset to training and retrieving the model outputs.

**Notebook:** [RDD2022_YOLO_Baseline_Training.ipynb](https://colab.research.google.com/drive/1llypLXIO6GG-HlhJQOhKxhS8--BVx03h)

## 1. What the notebook does

The notebook prepares the RDD2022 road-damage dataset in YOLO format, creates a smaller class-balanced training subset, and trains lightweight object-detection models for comparison.

The notebook includes these main stages:

1. Mount Google Drive and extract the dataset archive.
2. Read the YOLO-formatted images and labels.
3. Create an initial subset for inspection/experimentation.
4. Build a class-balanced subset and generate its `dataset.yaml` configuration.
5. Train a YOLOv8 Nano model.
6. Train a YOLOv5 Nano model for comparison.
7. Evaluate/inspect results and prepare a model for deployment, where the relevant notebook cells are provided.

The nine class IDs used by the notebook are:

| ID | Class |
|---:|---|
| 0 | D00 |
| 1 | D01 |
| 2 | D10 |
| 3 | D11 |
| 4 | D20 |
| 5 | D40 |
| 6 | D43 |
| 7 | D44 |
| 8 | D50 |

## 2. Prepare Google Drive

Before running the notebook:

1. Make sure you can access the notebook with the Google account that owns or has permission to use it.
2. Upload the compressed YOLO dataset archive to the **root of My Drive**.
3. Name the archive exactly:

   `RDD2022_YOLO.tar.gz`

The first notebook cell expects this exact location:

```text
/content/drive/MyDrive/RDD2022_YOLO.tar.gz
```

If your archive is inside a Drive folder, edit `archive_path` in the first cell to point to its actual location. For example:

```python
archive_path = '/content/drive/MyDrive/datasets/RDD2022_YOLO.tar.gz'
```

The archive is expected to contain a YOLO dataset directory with `images/` and `labels/` folders, each containing `train/` and `val/` splits. The later code expects paths equivalent to:

```text
RDD2022_YOLO/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Do not assume the archive is correct just because extraction completes. Verify that these folders exist.

## 3. Open the notebook and configure Colab

1. Open the notebook using the link above.
2. If prompted, sign in to Google.
3. In Colab, select **Runtime → Change runtime type**.
4. Choose **T4 GPU** under Hardware accelerator, if GPU access is available to your account.
5. Save the runtime settings and connect to the runtime.

A T4 option being displayed or selected does not, by itself, prove that the training process is using a GPU. After connecting, run this diagnostic cell:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No CUDA GPU detected. Training will run on CPU.")
```

For a GPU run, `torch.cuda.is_available()` should print `True` and the GPU name should be shown. If it prints `False`, check the runtime type and reconnect. CPU training can work, but it may take substantially longer.

## 4. Execute the notebook in order

Run the notebook from top to bottom. Do not skip preparation cells unless you have confirmed that their outputs already exist in the current runtime.

In Colab, click the play icon beside a cell, or select **Runtime → Run all**. Running a cell executes only that cell; running all executes the notebook sequentially.

### Stage A — Mount Drive and extract the archive

Run the first cell. It:

- Mounts Google Drive at `/content/drive`.
- Checks for `RDD2022_YOLO.tar.gz`.
- Extracts the archive into `/content/RDD2022_YOLO`.

When Google asks for permission, authorize Drive access.

Expected messages include:

```text
Mounted at /content/drive
Extracting RDD2022_YOLO.tar.gz to /content/ ...
Extraction complete!
```

If the archive is not found, stop and correct `archive_path`. Do not continue to the data-preparation cells until extraction succeeds.

You can verify the extracted structure with:

```python
from pathlib import Path

root = Path('/content/RDD2022_YOLO')

for path in [
    root / 'images' / 'train',
    root / 'images' / 'val',
    root / 'labels' / 'train',
    root / 'labels' / 'val',
]:
    print(path, "exists:", path.exists())
```

All four checks should be `True`.

### Stage B — Create the initial subset

Run the cell that defines and calls `build_subset(...)`.

The code reads labelled images from the original train and validation splits, selects images with bounding-box annotations, and creates:

```text
/content/RDD2022_Subset/
```

The notebook's current call uses:

- `images_per_class=350`
- `val_images=1000`
- `seed=42`

This is an intermediate subset-generation step. It is **not** the dataset path used by the later YOLOv8 training cell. The later training cell points to the balanced subset described below.

### Stage C — Build the class-balanced subset

Run the section headed **CLASS BALANCING**.

This cell searches the source `images/` and `labels/` trees, matches image files to YOLO `.txt` labels by filename stem, limits the number of retained label instances per class, shuffles the selected image-label pairs, and creates an 80/20 train-validation split.

The current settings include:

```python
source_dir = Path('/content/RDD2022_YOLO')
target_dir = Path('/content/RDD2022_YOLO_Balanced_Subset')
MAX_INSTANCES_PER_CLASS = 600
```

The resulting dataset is written to:

```text
/content/RDD2022_YOLO_Balanced_Subset/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The cell also writes `dataset.yaml`, which tells Ultralytics where the images are and maps class IDs to class names.

**Important:** The class-balancing cell deletes `target_dir` if it already exists, then recreates it. If you have files in that directory that you need, copy them elsewhere before rerunning the cell.

The cell matches labels to images by filename stem and keeps the first image found for a stem. If the dataset contains duplicate stems in different subfolders, inspect the matching logic to ensure it does not pair a label with the wrong image.

After this cell completes, verify that the dataset configuration exists:

```python
from pathlib import Path

yaml_path = Path('/content/RDD2022_YOLO_Balanced_Subset/dataset.yaml')
print("dataset.yaml exists:", yaml_path.exists())

if yaml_path.exists():
    print(yaml_path.read_text())
```

Do not start training if the file is missing or if its paths do not point to the generated dataset.

### Stage D — Clean up memory

Run the cleanup cell that imports `gc` and `torch`. It collects unused Python objects and clears the CUDA cache when a GPU is available.

This is a memory-management step; it does not train a model.

### Stage E — Train YOLOv8 Nano

Run the cell that loads `YOLO('yolov8n.pt')` and calls `.train(...)`.

The notebook currently uses these key settings:

| Setting | Value |
|---|---|
| Model | `yolov8n.pt` |
| Dataset | `/content/RDD2022_YOLO_Balanced_Subset/dataset.yaml` |
| Epochs | 25 |
| Image size | 640 |
| Batch size | 16 |
| Workers | 2 |
| Image cache | Disabled |
| Run name | `Compare_YOLOv8n_Balanced` |

Training results are directed to:

```text
/content/drive/MyDrive/RDD2022_Comparisons/Compare_YOLOv8n_Balanced/
```

Because the project directory is on Google Drive, the results should persist after the Colab runtime is disconnected. Training data is read from `/content`, which is temporary Colab storage.

The training cell may take a long time, particularly if Colab is using CPU rather than GPU. Allow it to finish. Do not close the tab or interrupt the runtime while an epoch is running unless you intentionally want to stop training.

### Stage F — Train YOLOv5 Nano

Run the subsequent YOLOv5 comparison cell. It loads:

```python
YOLO('yolov5nu.pt')
```

and trains for 25 epochs using the same balanced dataset configuration.

Its output is directed to the comparisons folder in Google Drive, under the run name:

```text
Compare_YOLOv5nu_Balanced
```

The notebook currently sets `batch=32` for this run. If you encounter an out-of-memory error, reduce the batch size (for example, to 16 or 8) and rerun the training cell.

Run the two model-training cells separately and confirm that each completes. Do not compare models using incomplete training runs.

## 5. Monitor and understand training

Ultralytics prints training progress for each epoch. Important metrics include:

- **Precision (P):** Of the detections made by the model, how many were correct.
- **Recall (R):** Of the labelled objects, how many the model detected.
- **mAP50:** Mean average precision at an IoU threshold of 0.50.
- **mAP50-95:** Mean average precision averaged over IoU thresholds from 0.50 to 0.95. This is stricter and usually lower.
- **Box loss:** Bounding-box localisation error.
- **Class loss:** Classification error.
- **DFL loss:** Distribution focal loss used in box localisation.

Do not judge the model from one early epoch. Wait for training and validation to finish, then compare the final and best-checkpoint metrics. A successful run is not automatically a good model: inspect per-class performance, especially for rare classes such as D01 and D11.

## 6. Find and download the outputs

Open Google Drive and navigate to:

```text
My Drive/
└── RDD2022_Comparisons/
    ├── Compare_YOLOv8n_Balanced/
    └── Compare_YOLOv5nu_Balanced/
```

Ultralytics normally saves run artifacts such as:

- `weights/best.pt` — checkpoint with the best validation fitness during training.
- `weights/last.pt` — checkpoint from the final epoch.
- `results.csv` — metrics recorded across epochs.
- `results.png` — training and validation plots.
- `confusion_matrix.png` and related validation plots, when generated.
- `args.yaml` — training configuration.

The exact files depend on whether each run completed and which plots were enabled. Confirm that the `weights/` directory exists before treating a run as complete.

Download `best.pt` for the model you intend to evaluate or deploy. Keep the run's `results.csv`, plots, and configuration as experimental evidence.

## 7. Runtime interruptions and resuming work

Colab runtimes are temporary. Files under `/content/` are normally removed when the runtime is terminated or reset. Files saved under `/content/drive/MyDrive/` persist in Google Drive.

If the runtime disconnects:

1. Reconnect to a runtime.
2. Rerun the Drive-mount and extraction cells.
3. Rerun the subset-generation cells if their `/content/` outputs are no longer present.
4. Check the relevant training output folder in Drive before starting training again.

The notebook's YOLOv8 call uses `exist_ok=True`, but that setting is not the same as resuming training. The shown call does not set `resume=True`. Do not assume rerunning the cell will continue from the last checkpoint. To resume an interrupted Ultralytics run, explicitly load the appropriate `last.pt` checkpoint and use the supported resume workflow, after checking the run directory and Ultralytics version.

## 8. Common problems

| Problem | What to check |
|---|---|
| Archive not found | Confirm the file is in My Drive and that `archive_path` is correct. |
| Extraction fails | Confirm the archive is a valid `.tar.gz` file and that it contains the expected dataset structure. |
| `images/train` or `labels/train` missing | Inspect the archive's internal folder structure and adjust extraction/path logic if necessary. |
| `dataset.yaml` missing | Rerun the class-balancing cell and inspect any error output. |
| No matching image-label pairs | Check that image and label filenames share the same stem and that the source folders are correct. |
| CUDA unavailable | Recheck **Runtime → Change runtime type**, reconnect, and run the CUDA diagnostic cell. |
| Out-of-memory error | Reduce `batch` (e.g. 16 or 8), use a smaller model, or obtain a runtime with more memory. |
| Training is extremely slow | Check whether the runtime is using CPU. The notebook's visible training output previously reported a CPU device, despite a T4 option being shown in the interface. |
| Runtime disconnected | Reconnect and rebuild temporary `/content/` data; check Drive for saved outputs before retraining. |
| Existing output directory causes confusion | Inspect the run folder and checkpoint timestamps before rerunning; do not overwrite results you need to preserve. |

## 9. Important experimental checks before reporting results

There are a few issues worth checking rather than blindly accepting the notebook output:

1. **Confirm the actual runtime device.** A training log that says `CPU` means that run did not use the GPU. Record the device used for each experiment.
2. **Check the validation split.** The balancing cell randomly splits selected image-label pairs 80/20. Confirm that the split is appropriate for your experiment and that no near-duplicate images or related sequences leak across train and validation.
3. **Check class balancing carefully.** The code caps retained *instances* per class, not necessarily the number of images per class. Images can contain multiple classes, and some classes may still have far fewer examples.
4. **Check label integrity.** Confirm that class IDs are between 0 and 8 and each YOLO label line contains five numeric values: `class_id x_center y_center width height`. The four box values should be normalised to the range 0–1.
5. **Keep the experiment reproducible.** Record the dataset version, subset settings, random seed, model weights, Ultralytics/PyTorch versions, runtime hardware, epochs, batch size, image size, and final metrics.
6. **Do not treat a short baseline as deployment-ready.** Twenty-five epochs on a sampled subset is a baseline experiment. It is not sufficient evidence on its own that the model will perform reliably on Kenyan roads or on mobile devices.

## 10. Recommended execution checklist

- [ ] Dataset archive uploaded to Google Drive.
- [ ] Archive path in the first cell checked.
- [ ] Colab runtime connected and hardware type selected.
- [ ] CUDA diagnostic run and device recorded.
- [ ] Drive mounted and dataset extracted successfully.
- [ ] Expected image and label directories verified.
- [ ] Initial subset cell completed.
- [ ] Class-balanced subset created.
- [ ] `dataset.yaml` checked.
- [ ] YOLOv8 Nano training completed.
- [ ] YOLOv5 Nano training completed.
- [ ] Validation metrics and plots inspected.
- [ ] `best.pt` checkpoints and experiment evidence confirmed in Drive.
- [ ] Runtime hardware and experiment settings documented.

---

**Notebook link:** https://colab.research.google.com/drive/1llypLXIO6GG-HlhJQOhKxhS8--BVx03h

**Note:** This guide follows the notebook's current paths and parameter values. If the notebook is edited, update this guide to match the new cells and settings.

# RDD2022 Exploration Setup Guide

This project is a Python notebook used to explore the Road Damage Dataset 2022 (RDD2022). It scans image files, matches them with XML annotation files, and builds a DataFrame for analysis.

## Project contents

- `RDD2022_Exploration.ipynb` — the main notebook containing the data-loading and analysis code
- `.venv/` — the project virtual environment

## Requirements

Before running the notebook, make sure you have:

- Python 3.9 or newer
- VS Code with the Python extension installed (recommended)
- Jupyter support enabled in VS Code
- The RDD2022 dataset available locally on your machine

## Recommended setup

### 1) Open the project folder

Open this folder in VS Code:

- `/home/topsy/3.2/AI/SemProject_MLEngine`

### 2) Use the included virtual environment

This workspace already contains a `.venv` folder, so you can use it instead of creating a new environment.

On Linux/macOS:

```bash
cd /home/topsy/3.2/AI/SemProject_MLEngine
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
cd "C:\path\to\SemProject_MLEngine"
.\.venv\Scripts\Activate.ps1
```

### 3) Install needed Python packages

The notebook imports `pandas`, `glob`, `os`, and `xml.etree.ElementTree`. If the environment is missing any packages, install them:

```bash
python -m pip install pandas jupyter ipykernel
```

If you are already using the project `.venv`, this installs the packages in that environment.

## Dataset setup

The notebook currently points to this dataset directory:

```python
DATASET_PATH = "/home/topsy/3.2/AI/RDD2022/RDD2022"
```

That means the dataset must exist at the path above and follow a structure similar to:

```text
/home/topsy/3.2/AI/RDD2022/RDD2022/
    country_name/
        image.jpg
        image.xml
```

If your dataset is stored somewhere else, update the `DATASET_PATH` variable in the first code cell before running the notebook.

## Run the notebook

### Option 1: in VS Code

1. Open `RDD2022_Exploration.ipynb`.
2. Make sure the notebook is using the project Python environment.
3. Run the cells from top to bottom.
4. The notebook will:
   - find all `.jpg` files recursively
   - locate matching `.xml` annotation files
   - extract image metadata
   - count damage objects
   - collect damage classes
   - build a pandas DataFrame called `df`

### Option 2: from the terminal

You can also run the notebook with Jupyter:

```bash
cd /home/topsy/3.2/AI/SemProject_MLEngine
source .venv/bin/activate
jupyter notebook
```

Then open `RDD2022_Exploration.ipynb` in the browser tab that opens.

## What the notebook does

The notebook is designed to answer questions such as:

- How many images are in the dataset?
- What columns are present?
- Are there missing values?
- What do the first and last rows look like?
- Which countries or labels appear in the XML annotations?

The core logic reads each image file, looks for a sibling `.xml` file, and extracts:

- filename
- country from the filename prefix
- image width/height/depth
- whether damage is present
- number of detected objects
- damage class names

## Troubleshooting

### Dataset path not found

If you get file-not-found errors, check that the dataset folder really exists and update `DATASET_PATH` in the notebook.

### Missing package errors

If Python says a module is missing, install it with:

```bash
python -m pip install pandas
```

### Notebook not using the correct interpreter

In VS Code:

1. Open the Command Palette.
2. Choose "Python: Select Interpreter".
3. Select the `.venv` environment for this project.

## Notes

This project is primarily exploratory and educational. It is not a full training pipeline yet; it focuses on dataset inspection and understanding the structure of the RDD2022 annotations before further model development.

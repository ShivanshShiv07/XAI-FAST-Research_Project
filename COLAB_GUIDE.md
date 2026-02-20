
# ⚡️ Running XAI-Guided FAST on Google Colab (Free GPU)

Since your local machine is slow, you can run this entire project on Google Colab to utilize their free GPUs (usually Tesla T4).

## Step 1: Prepare Your Project
1.  Ensure you have the `MiroThinker` folder (which contains `libs`, `data`, etc.).
2.  **Zip the `MiroThinker` folder**. excluding `.venv` if possible to save space (but it's fine if you include it, we just won't use it there).

## Step 2: Upload to Google Drive
1.  Go to [drive.google.com](https://drive.google.com).
2.  Upload `MiroThinker.zip` to your My Drive root (or a folder).
3.  Right-click the zip and "Extract" it, or ensure you have the unzipped `MiroThinker` folder available on Drive.

## Step 3: Open Google Colab
1.  Go to [colab.research.google.com](https://colab.research.google.com).
2.  Click **New Notebook**.
3.  **Enable GPU**: Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU` (or better) -> Save.

## Step 4: Run the Following Cells
Copy and paste these commands into code cells in Colab.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Navigate to Project (Adjust path if needed)
```python
import os
# Change 'MiroThinker' to wherever you uploaded it
path = "/content/drive/MyDrive/MiroThinker" 
os.chdir(path)
print(f"Current Directory: {os.getcwd()}")
```

### Cell 3: Install Dependencies
```bash
!pip install -r requirements.txt
```
*(Note: It might say 'RESTART SESSION' at the bottom. Click it if asked, then re-run Cell 2).*

### Cell 4: Run Real Experiment (GPU Accelerated)
```python
# This will be MUCH faster than your local CPU
!python libs/fast_anomaly_synthesis/experiment_runner_real.py
```

## Full Training (MVTec + BTAD)
Full training is long and may take multiple Colab sessions. The script below is resumable; you can re-run it and it will continue where it stopped.

### Cell A: Download MVTec (Full)
```python
!python libs/fast_anomaly_synthesis/download_mvtec_real.py
```

### Cell B: Download BTAD (Kaggle)
```python
!python libs/fast_anomaly_synthesis/download_btad_kaggle.py
```

### Cell C: Train Both Datasets (Sequential)
```python
!python libs/fast_anomaly_synthesis/train_all.py \
  --dataset both \
  --gpus "0," \
  --max_steps 80000 \
  --batch_size 2
```

Tips:
- Start with `--limit 2` to verify everything works.
- If you run out of time, just re-run the same command to resume.
- If you hit GPU OOM, reduce `--batch_size`.

## Step 5: Check Results
Once finished, go to your Google Drive folder (`MiroThinker/results`) to see:
- `figure_3_comparison.png`
- `real_experiment_gradient.png` (The novelty visualization)

# 🎮 YouTube Audio Processing & Category Classification (EGBI222 Group Project)

This project automates the workflow of downloading YouTube videos, extracting their audio, transcribing and translating the speech content, cleaning and merging results, and finally training a machine learning model to classify each video into categories based on translated text.

Originally developed as a group assignment for **EGBI222**, this notebook coordinates distributed workloads across multiple machines and integrates tools such as **Whisper**, **yt-dlp**, and **scikit-learn** to achieve end-to-end processing.

---

## 🚀 Features

* **Automated Dataset Retrieval** from Kaggle
* **Parallel YouTube Audio Downloading** using `yt-dlp` and `ThreadPoolExecutor`
* **Whisper-based Transcription & Translation** (supports multilingual input)
* **Data Validation & Cleaning** for high-quality transcripts
* **Multi-machine Processing** (split work by `MACHINE_ID`)
* **Progress Autosaving** and resume capability in case of runtime interruption
* **CSV Merge & Cleaning** to create a master dataset
* **Text-based Category Classification** using TF-IDF + Logistic Regression
* **Visualization** with confusion matrix for model evaluation

---

## 🧩 Project Structure

```
EGBI222_GroupProject/
├── youtube_data.csv                   # Dataset from Kaggle
├── audio/                             # Folder for downloaded .mp3 files
├── results/                           # Intermediate results per machine
├── Master/                            # Final merged + cleaned dataset
└── _main_groupproject_egbi222.ipynb   # Main processing script (Colab)
```

---

## ⚙️ Setup & Requirements

### Environment

* Python 3.8+
* Google Colab (recommended)
* GPU (for faster Whisper inference)

### Dependencies

```bash
pip install kaggle yt-dlp git+https://github.com/openai/whisper.git faster-whisper
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install google-generativeai deep-translator langdetect tqdm scikit-learn matplotlib pandas numpy
```

You’ll also need:

* `kaggle.json` credentials placed in `/root/.kaggle/kaggle.json`
* Access to Google Drive for input/output paths

---

## 📦 Workflow Overview

### 1. **Setup**

Mount Google Drive, install dependencies, and configure Kaggle + API access.

### 2. **Data Preparation**

* Download dataset from Kaggle (`cyberevil545/youtube-videos-data-for-ml-and-trend-analysis`).
* Load and filter YouTube videos between **1–10 minutes** duration.

### 3. **Distributed Work Assignment**

The dataset is divided by index modulo across multiple machines:

```python
TOTAL_MACHINES = 5
MACHINE_ID = 1  # Change per user
```

Each collaborator works on a different segment, producing a partial results file.

### 4. **Audio Download**

Parallelized using threads:

```python
max_workers = 5
```

Skips already-downloaded files and logs progress with emoji markers (✅ / ⚠️ / ⏩).

### 5. **Transcription & Translation**

* Uses **Faster Whisper** model (default: `"small"`).
* Each audio is:

  * Transcribed (`transcribe_one`)
  * Translated to English (`translate_one`)
* Invalid or empty sentences are dropped automatically.

### 6. **Progress Saving**

After each batch of ~10 rows, results are saved to:

```
/content/drive/MyDrive/EGBI222_Group Project_1/results/progress_<name>.csv
```

Resumable after crashes or disconnects.

### 7. **Merging & Cleaning**

All partial results (`processed_*.csv`) are merged:

* Duplicates removed
* Empty or missing `Translate`/`category` rows dropped
* Output → `/Master/Master_clean.csv`

### 8. **Machine Learning Model**

Final step trains a classifier:

* **Text Vectorization:** TF-IDF (bigrams, 20k features)
* **Model:** Logistic Regression (balanced class weights)
* **Accuracy:** ~51.5%
* **Evaluation:** Confusion Matrix visualization

---

## 📊 Model Parameters

| Parameter          | Value               |
| ------------------ | ------------------- |
| Test size          | 0.30                |
| TF-IDF ngram range | (1, 2)              |
| min_df             | 4                   |
| max_features       | 20,000              |
| Classifier         | Logistic Regression |
| Balanced weights   | ✅                   |
| Max iterations     | 1000                |

---

## 💡 Notes & Recommendations

* Adjust `MACHINE_ID` and `SUFFIX` when multiple users process data.
* Whisper model size can be changed (`tiny`, `base`, `small`, `medium`, `large`) for trade-offs between speed and accuracy.
* For large-scale runs, ensure sufficient disk space (~10 GB per 2,000 clips).
* Use Google Drive autosync to prevent data loss.

---

## 📈 Example Output

```
Accuracy: 0.515
Classification Report:
              precision    recall  f1-score   support
  Music          0.53      0.50      0.51       140
  Gaming         0.55      0.52      0.53       145
  Vlog           0.49      0.47      0.48       136
  ...
```

Confusion matrix visualized with `matplotlib` for interpretability.

---

## 🧬 Future Improvements

* Try multilingual embeddings (e.g., `LaBSE`, `mBERT`)
* Use sentence transformers for semantic classification
* Integrate automatic language detection before transcription
* Build a web dashboard for dataset and model management

---

## 👥 Contributors

* **TZU-WEI CHEN            6713358**
* **Thanakorn Pechmanee     6713365**
* **Prem Paksin             6713378**
* **Peerapat Virojsirasak   6713384**
* **Pawach Takpiman         6713386**
* **Suphakorn Daengpayon    6713423**

---

## 📄 License

This project was developed for academic purposes (EGBI222 Group Project).
You are free to reuse the methodology with attribution.

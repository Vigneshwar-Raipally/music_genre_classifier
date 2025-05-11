# 🎵 Music Genre Classifier 🎶  
An interactive Streamlit app that predicts the genre of a `.wav` audio file using Machine Learning and MFCC-based audio feature extraction.

---

## 📁 Project Structure

```

music_genre_classifier/
│
├── app.py                       # Streamlit app for genre prediction
├── train_model.py              # Script to train model from extracted features
├── train_classifier.py         # Script to extract audio features from wav files
├── utils.py                    # Utility function for MFCC feature extraction
│
├── model/
│   └── music_genre_classifier.pkl   # Trained ML model (did not include as it was huge file)
│
├── features/
│   └── genre_features.csv          # Extracted audio features for training
│
├── data/
│   └── \[genre folders]/            # e.g., rock/, jazz/, classical/, pop/
│       └── \*.wav                   # WAV files (not included in repo due to size)
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

````

---

## 🔧 How It Works

1. Upload a `.wav` audio file
2. The app extracts MFCC features using `librosa`
3. The trained ML model (e.g., Random Forest / KNN / Gradient Boosting) predicts the music genre
4. Displays waveform and prediction confidence

---

## 🚀 Run the App Locally

### ✅ 1. Clone the repo
```bash
git clone https://github.com/yourusername/music_genre_classifier.git
cd music_genre_classifier
````

### ✅ 2. Create a virtual environment (optional but recommended)

```bash
python -m venv myenv
source myenv/bin/activate    # On Windows: myenv\Scripts\activate
```

### ✅ 3. Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🏗️ Train the Model from Scratch

If `music_genre_classifier.pkl` is not included:

### Step 1: Prepare folders

Place `.wav` files in respective genre folders inside `/data/`:

```
data/
├── rock/
├── classical/
├── jazz/
└── pop/
```

### Step 2: Extract features

```bash
python extract_features.py
```

### Step 3: Train model

```bash
python train_model.py
```

This will generate the model in `/model/music_genre_classifier.pkl`.

---

## 🌐 Deploy to Streamlit Cloud

---

## 🧠 Technologies Used

* Python, Streamlit
* Librosa (for audio processing)
* Scikit-learn (for model training)
* Matplotlib (for waveform visualization)

---

## Future Improvements:
- Support for more audio formats (e.g., MP3, FLAC).
- Improve the feature extraction process with additional techniques.

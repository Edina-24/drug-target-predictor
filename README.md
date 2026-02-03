# Drug-Target Interaction (DTI) Predictor 🧬

A Deep Learning tool to predict the binding affinity ($pK_d$) between a drug molecule (SMILES) and a target protein (Amino Acid Sequence).

## 🚀 Features
- **DeepDTA Architecture**: Two-branch Convolutional Neural Network (CNN) implemented in PyTorch.
- **Interactive UI**: User-friendly web interface built with Streamlit.
- **End-to-End Pipeline**: Includes data preprocessing, training script, and inference engine.

## 📂 Project Structure
```
/
├── src/
│   ├── app.py          # Streamlit Frontend & Inference Logic
│   ├── model.py        # PyTorch Model Architecture (DeepDTA)
│   ├── train.py        # Training Loop & Data Loading
│   └── preprocess.py   # Encoding Utils (Label Encoding)
├── data/               # (GitIgnored) Stores model.pth and datasets
├── requirements.txt    # Python Dependencies
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd drug-target
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### 1. Run the Web App
```bash
streamlit run src/app.py
```
Open your browser to `http://localhost:8501`.

### 2. Train the Model
To retrain the model (requires `PyTDC` or will use synthetic data mode):
```bash
python src/train.py
```

## 🧠 Model Details
- **Inputs**: 
  - Drug: SMILES string (Character-level embedding + 1D Conv)
  - Target: Protein Sequence (Amino Acid embedding + 1D Conv)
- **Output**: Continuous binding affinity score (pKd/pIC50).

## ⚠️ Notes
- The repository does not include the trained `model.pth` (it's large). You must run `python src/train.py` locally to generate it.
- Created for research demonstration purposes.

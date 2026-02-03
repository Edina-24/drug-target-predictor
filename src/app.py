import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys

# Ensure src modules are found
sys.path.append(os.path.dirname(__file__))

from model import DeepDTA
from preprocess import drug_encoder, target_encoder

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/model.pth')
MAX_DRUG_LEN = 100
MAX_TARGET_LEN = 1200

# Set page config
st.set_page_config(page_title="Drug-Target Interaction Predictor", layout="wide")

# Custom CSS for research-grade aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333;
        font-weight: 700;
        text-align: center;
    }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepDTA(drug_vocab_size=drug_encoder.vocab_size, 
                    target_vocab_size=target_encoder.vocab_size)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except RuntimeError:
             # In case of size mismatch if vocab changed, usually requires retraining.
             # Ideally we save metadata. For now, assuming match.
             st.warning("Model architecture mismatch or file corrupted. Using initialized weights (Testing Mode).")
    else:
        st.warning(f"Model weights not found at {MODEL_PATH}. Using initialized weights (Prediction will be random).")
    
    model.to(device)
    model.eval()
    return model, device

st.markdown("<h1 class='title-text'>Drug-Target Interaction Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Predict binding affinity (pKd) using Deep Learning</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧬 Drug Molecule (SMILES)")
    default_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    drug_input = st.text_area("Enter SMILES string", value=default_smiles, height=150)

with col2:
    st.markdown("### 🧬 Target Protein (Sequence)")
    # Default ABL1 sequence (truncated for display, but full logic handles it)
    default_seq = "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
    target_input = st.text_area("Enter Amino Acid Sequence", value=default_seq, height=150)

if st.button("Predict Affinity"):
    if not drug_input or not target_input:
        st.error("Please provide both Drug and Target inputs.")
    else:
        model, device = load_model()
        
        # Preprocess
        with st.spinner("Processing inputs..."):
            d_enc = drug_encoder.encode(drug_input, max_len=MAX_DRUG_LEN)
            t_enc = target_encoder.encode(target_input, max_len=MAX_TARGET_LEN)
            
            d_tensor = torch.LongTensor(d_enc).unsqueeze(0).to(device)
            t_tensor = torch.LongTensor(t_enc).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                prediction = model(d_tensor, t_tensor).item()
        
        # Display
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color: #4CAF50;">Predicted pKd: {prediction:.2f}</h2>
            <p style="font-size: 1.1em;">
                This value represents the negative log of the dissociation constant (Kd).<br>
                <b>Higher values indicate stronger binding affinity.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Research Use Only. Not for Clinical Use.*")

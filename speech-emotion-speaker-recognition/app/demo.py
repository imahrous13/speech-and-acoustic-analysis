

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predict import predict_emotion, predict_speaker
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Speech Detection",
    page_icon="🎤",
    layout="wide"
)


st.title("🎤 Speech Emotion Detection")
st.markdown("---")


mode = st.sidebar.selectbox("Mode", ["Emotion", "Speaker ID"], index=0)


models_dir = Path(__file__).parent.parent / 'models'
expected_speaker_files = [
    'speaker_model.pkl',
    'speaker_label_encoder.pkl',
    'speaker_scaler.pkl',
    'speaker_feature_names.pkl'
]
missing_speaker_artifacts = [f for f in expected_speaker_files if not (models_dir / f).exists()]
if missing_speaker_artifacts:
    st.sidebar.warning(
        "Speaker-ID unavailable — missing files: " + ", ".join(missing_speaker_artifacts)
    )
else:
    st.sidebar.info("Speaker-ID: ready (model artifacts found in `models/`).")





uploaded_file = st.file_uploader(
    "Upload an audio file (.wav format)",
    type=['wav', 'mp3', 'flac']
)

if uploaded_file is not None:
    
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / uploaded_file.name
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File uploaded: {uploaded_file.name}")
    
   
    
    btn_label = "🔍 Predict Emotion" if mode == "Emotion" else "🔍 Predict Speaker"
    disabled_btn = (mode == "Speaker ID" and len(missing_speaker_artifacts) > 0)
    if disabled_btn and mode == "Speaker ID":
        st.caption("Speaker-ID is unavailable because required model files are missing.")

    if st.button(btn_label, type="primary", disabled=disabled_btn):
        with st.spinner("Processing audio and extracting features..."):
            try:
                if mode == "Emotion":
                    result = predict_emotion(str(temp_file_path))
                else:
                    artifact_error = False
                    try:
                        result = predict_speaker(str(temp_file_path))
                    except FileNotFoundError as e:
                        st.error(str(e))
                        artifact_error = True
                        result = None

                if result is None:
                    
                    if not (mode == "Speaker ID" and 'artifact_error' in locals() and artifact_error):
                        st.error("Error: Could not process the audio file. Please check the file format or model artifacts.")
                else:
                    
                    st.markdown("---")
                    st.header("📊 Prediction Results")

                    
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        if mode == "Emotion":
                            label = "Predicted Emotion"
                            value = result.get('predicted_emotion', '')
                            if isinstance(value, str):
                                value = value.upper()
                        else:
                            label = "Predicted Speaker"
                            value = result.get('predicted_speaker', '')

                        st.metric(label=label, value=value)

                    with col2:
                        probs = result.get('probabilities')
                        if probs:
                            title = "Prediction Probabilities"
                            st.markdown("### " + title)

                            
                            col_name = 'Emotion' if mode == 'Emotion' else 'Speaker'
                            prob_df = pd.DataFrame(
                                list(probs.items()),
                                columns=[col_name, 'Probability']
                            )
                            prob_df = prob_df.sort_values('Probability', ascending=False)

                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = plt.cm.viridis(prob_df['Probability'].values)
                            bars = ax.barh(prob_df[col_name], prob_df['Probability'], color=colors)
                            ax.set_xlabel('Probability', fontsize=12)
                            ax.set_ylabel(col_name, fontsize=12)
                            ax.set_title(f'{col_name} Prediction Probabilities', fontsize=14, fontweight='bold')
                            ax.set_xlim(0, 1)

                            
                            for i, (lbl, prob) in enumerate(zip(prob_df[col_name], prob_df['Probability'])):
                                ax.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=10)

                            plt.tight_layout()
                            st.pyplot(fig)

                            
                            st.markdown("### Detailed Probabilities")
                            prob_df['Probability (%)'] = (prob_df['Probability'] * 100).round(2)
                            prob_df_display = prob_df[[col_name, 'Probability (%)']].copy()
                            st.dataframe(prob_df_display, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)
            finally:
                
                if temp_file_path.exists():
                    temp_file_path.unlink()
    
    
    st.markdown("---")
    st.header("🎵 Audio Player")
    st.audio(uploaded_file, format='audio/wav')

else:
    st.info("👆 Please upload an audio file to get started.")
    
   


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    
    "</div>",
    unsafe_allow_html=True
)


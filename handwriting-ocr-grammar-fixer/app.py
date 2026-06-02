"""Streamlit web UI for Handwriting OCR + Grammar Fixer."""

import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from preprocess import preprocess
from ocr_extract import extract_with_confidence
from grammar_fix import analyze, load_tool

st.set_page_config(
    page_title="Handwriting OCR + Grammar Fixer",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwriting OCR + Grammar Fixer")
st.markdown("Upload a handwritten image → extract text → get grammar-corrected output.")
st.divider()

@st.cache_resource
def get_tool():
    return load_tool()

uploaded = st.file_uploader(
    "Upload a handwritten image",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Clear, well-lit handwritten text works best"
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(tmp_path, use_column_width=True)

    with st.spinner("🔍 Preprocessing & extracting text..."):
        try:
            processed = preprocess(tmp_path)
            ocr_result = extract_with_confidence(processed)
            raw_text = ocr_result['text']
        except Exception as e:
            st.error(f"OCR Error: {e}")
            raw_text = ""

    with col2:
        st.subheader("📝 Preprocessed Image")
        st.image(processed, use_column_width=True, clamp=True)

    st.divider()

    if raw_text:
        with st.spinner("✨ Fixing grammar..."):
            tool = get_tool()
            grammar_result = analyze(raw_text)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📤 Raw OCR Text")
            st.text_area("", raw_text, height=200, key="raw")
            st.caption(f"OCR Confidence: {ocr_result['avg_confidence']}%")

        with col4:
            st.subheader("✅ Grammar-Corrected Text")
            st.text_area("", grammar_result['corrected'], height=200, key="corrected")
            st.caption(f"{grammar_result['num_corrections']} correction(s) applied")

        if grammar_result['corrections']:
            with st.expander(f"🔍 View {grammar_result['num_corrections']} correction(s)"):
                for c in grammar_result['corrections']:
                    st.markdown(f"- **`{c['error']}`** → `{c['suggestions'][:1]}` — _{c['message']}_")
    else:
        st.warning("⚠️ No text could be extracted. Try a clearer image.")

    os.unlink(tmp_path)
else:
    st.info("⬆️ Upload a handwritten image to get started.")

import streamlit as st
import numpy as np
import pickle
import os, time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Config ----------
MODEL_FILES = {
    "LSTM": "next_word_lstm.h5",
    "GRU":  "next_word_lstm_GRU.h5",
}

DEFAULT_PROMPT = "To be or not to"

# ---------- Caching ----------
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    return load_model(path)

@st.cache_resource(show_spinner=False)
def load_tokenizer(pkl_path: str = "tokenizer.pkl"):
    with open(pkl_path, "rb") as handle:
        return pickle.load(handle)

# ---------- Core ----------
def predict_next_word(model, tokenizer, text, maxseqlen):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= maxseqlen:
        token_list = token_list[-(maxseqlen - 1):]

    token_list = pad_sequences([token_list], maxlen=maxseqlen - 1, padding='pre')

    # prediction
    predicted = model.predict(token_list, verbose=0)

    # get index of top-1 prediction
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    # use tokenizer's index -> word mapping if available
    if hasattr(tokenizer, "index_word") and predicted_word_index in tokenizer.index_word:
        return tokenizer.index_word[predicted_word_index]

    # fallback to reverse search (slower)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

def model_metrics(model_name: str, model_path: str, sample_text: str, tokenizer):
    # load once (cached) to compute metrics
    model = load_keras_model(model_path)

    # Params & layers
    params = model.count_params()
    layers = len(model.layers)

    # File size (MB)
    try:
        fsize_mb = os.path.getsize(model_path) / (1024 * 1024)
    except OSError:
        fsize_mb = float("nan")

    # Inference latency (ms) over N runs
    N = 10
    # compute maxseqlen compatible with this model
    maxseqlen = (model.input_shape[1] or 0) + 1

    # warmup
    _ = predict_next_word(model, tokenizer, sample_text, maxseqlen)

    t0 = time.perf_counter()
    for _ in range(N):
        _ = predict_next_word(model, tokenizer, sample_text, maxseqlen)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / N

    return {
        "Model": model_name,
        "Parameters": f"{params:,}",
        "Layers": layers,
        "File size (MB)": f"{fsize_mb:.2f}",
        "Avg inference (ms)": f"{avg_ms:.1f}",
    }

# ---------- UI ----------
st.title('LSTM / GRU Next Word Prediction')

# Model choice
model_choice = st.radio("Choose model", list(MODEL_FILES.keys()), horizontal=True)

# Load tokenizer once
tokenizer = load_tokenizer()

# Text input
input_text = st.text_input("Enter the sequence of Words", DEFAULT_PROMPT)

# Show comparison metrics (both models) for quick reference
with st.expander("Model comparison (metrics)"):
    rows = []
    for name, path in MODEL_FILES.items():
        try:
            rows.append(model_metrics(name, path, input_text or DEFAULT_PROMPT, tokenizer))
        except Exception as e:
            rows.append({"Model": name, "Parameters": "—", "Layers": "—",
                         "File size (MB)": "—", "Avg inference (ms)": "—"})
            st.warning(f"Could not compute metrics for {name}: {e}")
    df = pd.DataFrame(rows)
    st.table(df)

# Predict button
if st.button('Predict Next Word'):
    # Load the selected model
    selected_model_path = MODEL_FILES[model_choice]
    model = load_keras_model(selected_model_path)

    # Determine max sequence length expected by the model
    maxseqlen = (model.input_shape[1] or 0) + 1

    next_word = predict_next_word(model, tokenizer, input_text, maxseqlen)
    st.write(f"**Model:** {model_choice}")
    st.write(f"Next Word: `{next_word}`")

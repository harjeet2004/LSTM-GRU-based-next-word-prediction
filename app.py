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
CORPUS_PATH = "hamlet.txt"         # <- replace with your real validation text if available
EVAL_SAMPLE_SIZE = 1000            # sequences per model for quick accuracy calculation
EVAL_BATCH_SIZE = 256

# ---------- Caching ----------
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    return load_model(path)

@st.cache_resource(show_spinner=False)
def load_tokenizer(pkl_path: str = "tokenizer.pkl"):
    with open(pkl_path, "rb") as handle:
        return pickle.load(handle)

@st.cache_data(show_spinner=False)
def load_corpus_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Core ----------
def predict_next_word(model, tokenizer, text, maxseqlen):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= maxseqlen:
        token_list = token_list[-(maxseqlen - 1):]

    token_list = pad_sequences([token_list], maxlen=maxseqlen - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    # Prefer tokenizer.index_word if present
    if hasattr(tokenizer, "index_word") and predicted_word_index in tokenizer.index_word:
        return tokenizer.index_word[predicted_word_index]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

def build_eval_arrays(tokenizer, text: str, maxseqlen: int, sample_size: int):
    """Create (X, y) pairs for next-word prediction from raw text."""
    tokens = tokenizer.texts_to_sequences([text])[0]
    if len(tokens) < maxseqlen:
        return None, None

    xs, ys = [], []
    # X: previous maxseqlen-1 tokens, y: next token
    for i in range(maxseqlen - 1, len(tokens)):
        x = tokens[i - (maxseqlen - 1): i]
        xs.append(x)
        ys.append(tokens[i])

    X = pad_sequences(xs, maxlen=maxseqlen - 1, padding='pre')
    y = np.array(ys, dtype=np.int32)

    # uniform subsample to keep fast
    if len(X) > sample_size:
        idx = np.linspace(0, len(X) - 1, num=sample_size, dtype=int)
        X = X[idx]
        y = y[idx]
    return X, y

def evaluate_model_accuracy(model_path: str, tokenizer, corpus_path: str,
                            sample_size: int = EVAL_SAMPLE_SIZE):
    """Return Top-1 acc, Top-5 acc, Perplexity computed on a lightweight eval set."""
    if not os.path.exists(corpus_path):
        return None

    model = load_keras_model(model_path)
    text = load_corpus_text(corpus_path)
    maxseqlen = (model.input_shape[1] or 0) + 1

    X, y = build_eval_arrays(tokenizer, text, maxseqlen, sample_size)
    if X is None or len(X) == 0:
        return None

    preds = model.predict(X, batch_size=EVAL_BATCH_SIZE, verbose=0)

    top1 = preds.argmax(axis=1)
    top1_acc = float((top1 == y).mean() * 100.0)

    # Top-5 accuracy
    # argpartition gets indices of 5 largest per row (unordered)
    top5 = np.argpartition(preds, -5, axis=1)[:, -5:]
    top5_acc = float(((top5 == y[:, None]).any(axis=1)).mean() * 100.0)

    # Perplexity: exp(average negative log-likelihood)
    eps = 1e-12
    true_probs = preds[np.arange(len(y)), y]
    cross_ent = -np.log(np.maximum(true_probs, eps)).mean()
    ppl = float(np.exp(cross_ent))

    return {"Top-1 (%)": top1_acc, "Top-5 (%)": top5_acc, "Perplexity": ppl}

def model_metrics(model_name: str, model_path: str, sample_text: str, tokenizer):
    model = load_keras_model(model_path)

    # Params & layers
    params = model.count_params()
    layers = len(model.layers)

    # File size (MB)
    try:
        fsize_mb = os.path.getsize(model_path) / (1024 * 1024)
    except OSError:
        fsize_mb = float("nan")

    # Latency (ms) over N runs
    N = 10
    maxseqlen = (model.input_shape[1] or 0) + 1
    # warmup
    _ = predict_next_word(model, tokenizer, sample_text, maxseqlen)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = predict_next_word(model, tokenizer, sample_text, maxseqlen)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / N

    # Accuracy (optional, only if corpus is available)
    eval_stats = evaluate_model_accuracy(model_path, tokenizer, CORPUS_PATH, EVAL_SAMPLE_SIZE)

    row = {
        "Model": model_name,
        "Parameters": f"{params:,}",
        "Layers": layers,
        "File size (MB)": f"{fsize_mb:.2f}",
        "Avg inference (ms)": f"{avg_ms:.1f}",
    }
    if eval_stats is not None:
        row.update({
            "Top-1 Acc (%)": f"{eval_stats['Top-1 (%)']:.2f}",
            "Top-5 Acc (%)": f"{eval_stats['Top-5 (%)']:.2f}",
            "Perplexity": f"{eval_stats['Perplexity']:.2f}",
        })
    else:
        row.update({"Top-1 Acc (%)": "—", "Top-5 Acc (%)": "—", "Perplexity": "—"})
    return row

# ---------- UI ----------
st.title('LSTM / GRU Next Word Prediction')

# Model choice
model_choice = st.radio("Choose model", list(MODEL_FILES.keys()), horizontal=True)

# Load tokenizer once
tokenizer = load_tokenizer()

# Text input
input_text = st.text_input("Enter the sequence of Words", DEFAULT_PROMPT)

# Comparison (now with accuracies)
with st.expander("Model comparison (metrics)"):
    rows = []
    for name, path in MODEL_FILES.items():
        try:
            with st.spinner(f"Evaluating {name}…"):
                rows.append(model_metrics(name, path, input_text or DEFAULT_PROMPT, tokenizer))
        except Exception as e:
            rows.append({"Model": name, "Parameters": "—", "Layers": "—",
                         "File size (MB)": "—", "Avg inference (ms)": "—",
                         "Top-1 Acc (%)": "—", "Top-5 Acc (%)": "—", "Perplexity": "—"})
            st.warning(f"Could not compute metrics for {name}: {e}")
    df = pd.DataFrame(rows)
    st.table(df)

# Predict button
if st.button('Predict Next Word'):
    selected_model_path = MODEL_FILES[model_choice]
    model = load_keras_model(selected_model_path)
    maxseqlen = (model.input_shape[1] or 0) + 1
    next_word = predict_next_word(model, tokenizer, input_text, maxseqlen)
    st.write(f"**Model:** {model_choice}")
    st.write(f"Next Word: `{next_word}`")

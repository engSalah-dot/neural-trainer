import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing  # ← مباشرة
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Trainer", page_icon="🧠", layout="wide")
st.title("🧠 Neural Network Trainer")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Hyperparameters")
learning_rate = st.sidebar.select_slider("Learning Rate",
    options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-3,
    format_func=lambda x: f"{x:.0e}")
batch_size = st.sidebar.select_slider("Batch Size",
    options=[32, 64, 128, 256, 512], value=256)
epochs = st.sidebar.slider("Epochs", 10, 200, 100, 10)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
use_bn = st.sidebar.checkbox("BatchNormalization", value=True)
optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
architecture = st.sidebar.multiselect("Hidden Layers",
    [256, 128, 64, 32, 16], default=[128, 64, 32, 16])

# ── Load Data ──────────────────────────────────────────────
st.subheader("📂 Data Source")
data_source = st.radio("", ["🏠 California Housing (جاهز)", "📁 Upload CSV"])

if data_source == "🏠 California Housing (جاهز)":
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["price"] = housing.target
    target_col = "price"
    st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    target_col = st.text_input("Target column name", value="price")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        st.info("⬆️ Upload a CSV file")
        st.stop()

with st.expander("👀 Preview Data"):
    st.dataframe(df.head(10), use_container_width=True)

# ── Prep Data ──────────────────────────────────────────────
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
y = df[target_col].values
split = int(0.8 * len(X))
X_train, X_valid = X[:split], X[split:]
y_train, y_valid = y[:split], y[split:]
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
std[std == 0] = 1
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std

# ── Build Model ────────────────────────────────────────────
def build_model(input_shape, architecture, dropout_rate, use_bn, lr, optimizer_name):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(architecture[0], input_shape=input_shape))
    if use_bn:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dropout(dropout_rate))
    for units in architecture[1:]:
        model.add(keras.layers.Dense(units))
        if use_bn:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation="linear"))
    opts = {
        "Adam": keras.optimizers.Adam(learning_rate=lr),
        "SGD": keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        "RMSprop": keras.optimizers.RMSprop(learning_rate=lr)
    }
    model.compile(loss="mse", optimizer=opts[optimizer_name], metrics=["mae"])
    return model

# ── Plot ───────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0E1117')
    for ax in axes:
        ax.set_facecolor('#262730')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
    axes[0].plot(history['loss'], label='Train Loss', color='#4C9BE8', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#E84C4C', linewidth=2, alpha=0.8)
    axes[0].set_title('📉 Loss Curve', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend(facecolor='#1E1E2E', labelcolor='white')
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(history['mae'], label='Train MAE', color='#4CE8A0', linewidth=2)
    axes[1].plot(history['val_mae'], label='Val MAE', color='#E8A04C', linewidth=2, alpha=0.8)
    axes[1].set_title('📊 MAE Curve', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend(facecolor='#1E1E2E', labelcolor='white')
    axes[1].grid(True, alpha=0.2)
    plt.tight_layout()
    return fig

# ── Train ──────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Train Model", type="primary", use_container_width=True):
    if not architecture:
        st.error("اختاري architecture الأول!")
        st.stop()

    model = build_model(X_train.shape[1:], architecture,
                        dropout_rate, use_bn, learning_rate, optimizer_name)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-6, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True, verbose=0)
    ]

    progress_bar = st.progress(0, text="Training...")
    history_data = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    chart_placeholder = st.empty()

    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress(min((epoch+1)/epochs, 1.0),
                text=f"Epoch {epoch+1}/{epochs} | loss: {logs['loss']:.4f} | val_loss: {logs['val_loss']:.4f}")
            history_data["loss"].append(logs["loss"])
            history_data["val_loss"].append(logs["val_loss"])
            history_data["mae"].append(logs["mae"])
            history_data["val_mae"].append(logs["val_mae"])
            if (epoch+1) % 5 == 0:
                fig = plot_history(history_data)
                chart_placeholder.pyplot(fig)
                plt.close()

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks + [StreamlitCallback()], verbose=0)

    progress_bar.progress(1.0, text="✅ Done!")

    st.subheader("🏆 Final Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train Loss", f"{history_data['loss'][-1]:.4f}")
    m2.metric("Val Loss",   f"{history_data['val_loss'][-1]:.4f}")
    m3.metric("Train MAE",  f"{history_data['mae'][-1]:.4f}")
    m4.metric("Val MAE",    f"{history_data['val_mae'][-1]:.4f}")

    fig = plot_history(history_data)
    st.pyplot(fig)
    plt.close()

    model.save("trained_model.keras")
    with open("trained_model.keras", "rb") as f:
        st.download_button("⬇️ Download Model", f,
            file_name="trained_model.keras", use_container_width=True)

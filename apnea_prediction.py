import numpy as np
import tensorflow as tf

DIR = "fetched_recordings"
PATIENT_ID = "00001463-100507"
EVENT = "apnea" # or non_apnea
SEGMENT = "seg_252"
MODEL_PATH = "models/best_model.keras"

def test_audio(filename, model):
    path = f"{DIR}/{PATIENT_ID}/{EVENT}/{SEGMENT}/{filename}"
    spec = preprocess_spec(path)
    prob, label = predict(model, spec)
    print(f"{filename}: Probability = {prob:.4f} -> Class: {label}")

def predict(model, spec):
    prob = model.predict(spec)[0][0]
    # if prob is > 0.5, then the event is apneic
    label = "Apnea" if prob > 0.5 else "Normal"
    return prob, label

def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(f"{model_path}")

def preprocess_spec(path):
    spec = np.load(path)
    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
    spec = np.expand_dims(spec, axis=-1)  # (64, 313, 1)
    spec = np.expand_dims(spec, axis=0)  # (1, 64, 313, 1)
    spec = spec.astype(np.float32)
    return spec

if __name__ == "__main__":
    model = load_model()

    # test original segment
    test_audio("original_spectrogram.npy", model)
    # test raw segment
    test_audio("recorded_spectrogram_raw.npy", model)
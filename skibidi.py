import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading

class TextPredictorAI:
    def __init__(self):
        self.max_sequence_len = 40
        self.vocab_size = 1000
        self.model = self.build_model()
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.train_data = []
        self.input_sequence = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.started_predicting = False

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=64, input_length=self.max_sequence_len),
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(64, activation='relu'),
            Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_text(self, text):
        text = text.lower()
        tokens = list(text)
        for char in tokens:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                if idx < self.vocab_size - 1:
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
                else:
                    return [self.char_to_idx.get(char, self.vocab_size - 1) for char in tokens]
        return [self.char_to_idx.get(char, self.vocab_size - 1) for char in tokens]

    def train_model(self, progress_callback=None):
        if len(self.train_data) >= 100:
            X = pad_sequences(self.train_data[:-1], maxlen=self.max_sequence_len)
            y = np.array([seq[-1] if len(seq) > 0 else 0 for seq in self.train_data[1:]])
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if progress_callback:
                        progress_callback((epoch + 1) / 5)  # Assuming 5 epochs

            self.model.fit(X, y, epochs=5, verbose=0, callbacks=[ProgressCallback()])
            self.train_data = []
            print("Model training completed")  # Debugging output

    def predict_next(self, text):
        sequence = self.preprocess_text(text)
        self.train_data.append(sequence)
        self.input_sequence = (self.input_sequence + sequence)[-self.max_sequence_len:]
        
        if len(self.input_sequence) < self.max_sequence_len:
            return ''

        padded_input = pad_sequences([self.input_sequence], maxlen=self.max_sequence_len)
        prediction = self.model.predict(padded_input, verbose=0)
        predicted_idx = np.argmax(prediction, axis=-1)[0]
        predicted_char = self.idx_to_char.get(predicted_idx, '')
        print(f"Predicted character: {predicted_char}")  # Debugging output
        
        if predicted_char:
            self.started_predicting = True
        
        if self.started_predicting:
            self.total_predictions += 1
            if predicted_char == text[-1]:
                self.correct_predictions += 1
        
        return predicted_char

    def get_accuracy(self):
        if self.total_predictions == 0:
            return 0
        return (self.correct_predictions / self.total_predictions) * 100

class TextPredictorApp:
    def __init__(self, root, ai):
        self.root = root
        self.ai = ai
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Real-Time Text Predictor with AI")
        self.root.geometry("600x500")
        self.root.configure(bg="lightgray")

        self.label = tk.Label(self.root, text="Start typing below:", font=("Arial", 14), bg="lightgray")
        self.label.pack(pady=10)

        self.textbox = tk.Text(self.root, font=("Arial", 14), wrap='word', height=10, width=50)
        self.textbox.pack(pady=10)

        self.prediction_label = tk.Label(self.root, text="Prediction: ", font=("Arial", 12), bg="lightgray")
        self.prediction_label.pack(pady=5)

        self.accuracy_label = tk.Label(self.root, text="Accuracy: 0%", font=("Arial", 12), bg="lightgray")
        self.accuracy_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.textbox.bind("<KeyRelease>", self.on_key_release)

    def on_key_release(self, event):
        text = self.textbox.get("1.0", tk.END).strip()
        if text:
            prediction = self.ai.predict_next(text)
            self.prediction_label.config(text=f"Prediction: {prediction}")
            self.accuracy_label.config(text=f"Accuracy: {self.ai.get_accuracy():.2f}%")
            threading.Thread(target=self.train_model_thread, args=(text,)).start()

    def train_model_thread(self, text):
        self.ai.train_model(self.update_progress)
        self.root.after(0, self.reset_progress_bar)

    def update_progress(self, value):
        self.root.after(0, self._update_progress, value)

    def _update_progress(self, value):
        self.progress_bar["value"] = value * 100
        self.root.update_idletasks()

    def reset_progress_bar(self):
        self.progress_bar["value"] = 0
        self.root.update_idletasks()

if __name__ == "__main__":
    ai = TextPredictorAI()
    root = tk.Tk()
    app = TextPredictorApp(root, ai)
    
    # Add the label with the text "made by skibid akaash" at the bottom left
    label = tk.Label(root, text="made by skibid akaash")
    label.pack(side=tk.LEFT, anchor=tk.SW)
    root.mainloop()
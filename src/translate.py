import tensorflow as tf
import pickle
import numpy as np
import os


checkpoints_file = '../model/checkpoints.txt'

if os.path.exists(checkpoints_file) and os.path.getsize(checkpoints_file) > 0:
    with open(checkpoints_file, 'r') as f:
        lines = f.readlines()
        latest_model_path = lines[-1].strip()  # Remove leading/trailing whitespace and newlines if any
        # Optionally, check if the path exists or is valid before using it
        if not os.path.exists(latest_model_path):
            raise FileNotFoundError(f"Model path '{latest_model_path}' from checkpoints file does not exist.")
else:
    raise FileNotFoundError(f"Checkpoints file '{checkpoints_file}' does not exist or is empty.")

# ======================================================================================================

try:
    model = tf.keras.models.load_model(latest_model_path)
except:
    raise Exception("Error loading the model!")

# Load the vocabulary dictionary for both src and tgt

try:
    with open('../data/src_vocab.pkl', 'rb') as f:
        src_vocab_mapping = pickle.load(f)

    with open('../data/tgt_vocab.pkl', 'rb') as f:
        tgt_vocab_mapping = pickle.load(f)
except:
    raise Exception("Error loading vocabs, train the model first")

def convert_int_to_text(int_sequences, vocab_map):
    # Create a reverse mapping from index to word
    reverse_vocab_map = {index: word for word, index in vocab_map.items()}
    text_sequences = []
    for int_sequence in int_sequences:
        text_sequence = []
        for token in int_sequence:
            if token in reverse_vocab_map:
                text_sequence.append(reverse_vocab_map[token])
        text_sequences.append(' '.join(text_sequence))
    return text_sequences


def translate(source_text, sequence_length=18):
    int_tokens = []
    padded_tokens = []
    
    # Convert source text to integer tokens, handling OOV words
    for word in source_text.split():
        word_lower = word.lower()
        if word_lower in src_vocab_mapping:
            int_tokens.append(src_vocab_mapping[word_lower])
        else:
            return "Out of vocabulary words"
    
    # Add padding tokens for maintaining the sequence length
    if len(int_tokens) < sequence_length:
        int_tokens = int_tokens + [0] * (sequence_length - len(int_tokens))
    else:
        int_tokens = int_tokens[:sequence_length]
    
    # SRC for encoder
    encoder_inputs = np.array([int_tokens])
    
    # Decoder input with <start> token
    decoder_input_sequence = np.zeros((1, sequence_length))
    decoder_input_sequence[0, 0] = tgt_vocab_mapping['<start>']

    translated_tokens = []

    # Autoregressive Decoding
    for i in range(1, sequence_length):
        predictions = model.predict([encoder_inputs, decoder_input_sequence])
        predicted_index = np.argmax(predictions[0, i - 1, :])
        translated_tokens.append(predicted_index)
        
        # Check if <end> token is predicted or if we exceed sequence length
        if predicted_index == tgt_vocab_mapping['<end>'] or i == sequence_length - 1:
            break
        
        decoder_input_sequence[0, i] = predicted_index
    
    # Convert translated tokens back to text
    translation = ''.join(convert_int_to_text([translated_tokens], tgt_vocab_mapping))
    return translation


while True:
    input_text = input("Enter source text:")

    if input_text == 'q':
        break

    translation = translate(input_text)
    print(translation)


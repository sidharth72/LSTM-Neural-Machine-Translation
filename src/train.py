import numpy as np
from model import build_nmt_model
from preprocess import preprocess
import os
import pickle
from datetime import datetime

def train(
    src, 
    tgt, 
    labels, 
    sequence_length, 
    embedding_dim, 
    latent_dim, 
    src_vocab_size, 
    tgt_vocab_size,
    batch_size,
    epochs):

    lstm_model = build_nmt_model(
        sequence_length = sequence_length, 
        embedding_dim = embedding_dim, 
        latent_dim = latent_dim, 
        src_vocab_size = src_vocab_size, 
        tgt_vocab_size = tgt_vocab_size,
    )

    history = lstm_model.fit([src, tgt], labels, batch_size = batch_size, epochs = epochs)

    return lstm_model, history


def save_model(model, model_dir: str = '../model'):
    """
    Save a Keras model to a specified directory.

    Args:
        model (tf.keras.models.Model): Pretrained Keras model to be saved.
        model_dir (str): Directory path where the model will be saved. Default is '../model'.

    Returns:
        None
    """
    # Create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Generate a timestamp to make each save unique
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the model
    model_path = os.path.join(model_dir, f'model_{timestamp}.keras')
    model.save(model_path)
    print(f"Model saved successfully at {model_path}")

    with open('../model/checkpoints.txt', 'w') as f:
        f.write(model_path)
        print(f"checkpoints.txt updated")


# ====================================================================================================

csv_path = input("Enter your CSV file path containing src and tgt sentences:")
epochs = int(input("Enter the number of steps to train (Recommended: 10 - 20):"))


try:
    print("Started Preprocessing...")

    encoder_inputs, decoder_targets, decoder_lables, src_vocab, tgt_vocab = preprocess(
        csv_path=csv_path,
    )
except:
    raise Exception("Something wrong with the dataset, ensure the dataset is in the correct format")


try:
    print("Preprocessing completed! Training the model")

    lstm_model, history = train(
        encoder_inputs, 
        decoder_targets,
        decoder_lables,
        18,
        64,
        128,
        len(src_vocab),
        len(tgt_vocab),
        batch_size = 16,
        epochs = epochs
    )

    print(history)
    save_model(lstm_model)
    print("Trainig completed successfully...")
    with open('../data/src_vocab.pkl', 'wb') as f:
        pickle.dump(src_vocab, f)

    with open('../data/tgt_vocab.pkl', 'wb') as f:
        pickle.dump(tgt_vocab, f)

except:
    raise Exception("Training failed")

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
import tensorflow as tf

def build_nmt_model(sequence_length, embedding_dim, latent_dim, src_vocab_size, tgt_vocab_size):
    # Encoder
    encoder_input = Input(shape=(sequence_length,), name='encoder_inputs')
    encoder_embedding = Embedding(input_dim=src_vocab_size + 1, 
                                  output_dim=embedding_dim, 
                                  input_length=sequence_length, 
                                  trainable=True)(encoder_input)
    encoder_lstm, state_h, state_c = LSTM(latent_dim, 
                                          return_state=True, 
                                          return_sequences=True, 
                                          name='encoder_lstm')(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_input = Input(shape=(sequence_length,), name='decoder_inputs')
    decoder_embedding = Embedding(input_dim=tgt_vocab_size + 1, 
                                  output_dim=embedding_dim, 
                                  input_length=sequence_length, 
                                  trainable=True)(decoder_input)
    decoder_lstm = LSTM(latent_dim, 
                        return_sequences=True, 
                        return_state=True, 
                        name='decoder_lstm')
    decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention Layer
    attention = Attention(name='attention_layer')
    attention_out = attention([decoder_lstm_outputs, encoder_lstm])

    # Concatenate the attention outputs and the decoder outputs
    decoder_concat_input = tf.concat([decoder_lstm_outputs, attention_out], axis=-1)

    # Output layer
    decoder_dense = Dense(tgt_vocab_size + 1, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Model
    model = Model([encoder_input, decoder_input], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

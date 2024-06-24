# LSTM Neural Machine Translation - En to Hi

This repository contains an implementation of an LSTM encoder-decoder model with an attention mechanism for translating sentences from English to Hindi. The attention mechanism enhances the model's ability to focus on the important parts of the sequence during translation, leading to more accurate results.

## Model Architecture

The model is based on the encoder-decoder architecture with Long Short-Term Memory (LSTM) networks. The attention mechanism is applied to the encoder's hidden states to compute a context vector for each time step in the decoder. This context vector helps the decoder to focus on relevant parts of the input sequence when generating each word in the output sequence.

### Components:
- **Encoder**: An LSTM network that processes the input sequence and produces a sequence of hidden states.
- **Attention Mechanism**: Calculates attention scores for each hidden state of the encoder to generate a context vector.
- **Decoder**: An LSTM network that uses the context vector and previously generated words to produce the output sequence.

## Dataset

The model is trained on a dataset containing 100,000 pairs of English and Hindi sentences. This dataset provides a diverse range of sentence structures and vocabulary, contributing to the robustness of the translation model.

## Results

The results are good but not great on several arbitrarily chosen translations. Below are some examples:

![image](https://github.com/sidharth72/LSTM-Neural-Machine-Translation/assets/74226199/fa454746-d4d3-45de-b486-69cab5d3c58c)




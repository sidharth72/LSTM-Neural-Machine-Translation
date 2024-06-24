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

## Repository Structure

- `data/`: Contains dataset and vocabulary files.
- `src/`: Source code for the model implementation.
  - `model.py`: LSTM implementation.
  - `preprocess.py`: Script for performing data preprocessing.
  - `train.py`: Script for training the model.
  - `translate.py`: Script for translating sentences using the trained model.
- `notebooks/`: Jupyter notebooks for data exploration and model evaluation.
- `models/`: Directory to store results and model checkpoints.

## How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sidharth72/LSTM-Neural-Machine-Translation.git
    cd LSTM-Neural-Machine-Translation
    ```

2. **Navigate to the source directory:**
    ```bash
    cd src
    ```

3. **Upload your dataset:**
   Place your CSV dataset containing the source (`src`) and target (`tgt`) sentences in the `data/` folder. The dataset should only contain these two columns.

4. **Train the model:**
    ```bash
    python train.py
    ```
    Follow the prompts:
    - Enter your CSV file path containing `src` and `tgt` sentences: `../data/dataset.csv`
    - Enter the number of steps to train (Recommended: 10 - 20): `10`

   Training will start.

5. **Evaluate the model:**
    ```bash
    python translate.py
    ```
    This script will use the most recently trained model for translation.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the open-source community for providing the resources and tools that made this project possible.

---

Feel free to reach out with any questions or feedback.

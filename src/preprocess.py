import numpy as np
import pandas as pd
import re


def preprocess(csv_path, src_column_name, tgt_column_name):

    sequence_length = 18

    df = pd.read_csv(csv_path)

    # Removing nll columns
    if df.isnull().values.any():
        df = df.dropna()

    # Helper function for creating buckets
    def create_buckets(df, seq_length, tolerance_ratio=1.2):
        """
        Create buckets of sequences of constant size for machine translation.

        Args:
            df (pd.DataFrame): Input dataframe with source and target columns.
            seq_length (int): Desired sequence length for each bucket.
            tolerance_ratio (float): Tolerance ratio for the difference in sequence length between source and target.

        Returns:
            pd.DataFrame: Modified dataframe with buckets of the specified sequence length.
        """
        src_buckets = []
        tgt_buckets = []

        # Iterate over the dataframe rows
        for idx, row in df.iterrows():
            src_tokens = row[src_column_name].split()
            tgt_tokens = row[tgt_column_name].split()

            src_len = len(src_tokens)
            tgt_len = len(tgt_tokens)

            i = 0
            while i < max(src_len, tgt_len):
                src_bucket = src_tokens[i:i+seq_length]
                tgt_bucket = tgt_tokens[i:i+int(seq_length * tolerance_ratio)]

                # Append the bucket to the list
                src_buckets.append(' '.join(src_bucket))
                tgt_buckets.append(' '.join(tgt_bucket))

                # Move to the next bucket
                i += seq_length

        # Create the new dataframe
        bucketed_df = pd.DataFrame({
            src_column_name: src_buckets,
            tgt_column_name: tgt_buckets
        })

        return bucketed_df

    df = create_buckets(df, sequence_length)


    # Removing special characters and numbers
    def preprocess_source(sentence):
        """Function to remove special characters"""
        sentence = sentence.lower()
        sentence = re.sub(r'\d+', '', sentence)
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        sentence = sentence.strip()
        return sentence

    def preprocess_target(sentence):
        """Function to add start and end tokens"""
        sentence = '<start> ' + sentence + ' <end>'
        return sentence

    df[src_column_name] = df[src_column_name].apply(preprocess_source)
    df[tgt_column_name] = df[tgt_column_name].apply(preprocess_target)

    src_sentences = df[src_column_name].tolist()
    tgt_sentences = df[tgt_column_name].tolist()

    src_vocab_mapping = dict()
    tgt_vocab_mapping = dict()

    def generate_vocab_map(d, sentences):
        """Function to map unique integer tokens to each word in sentences"""
        index = 1
        for sentence in sentences:
            for word in sentence.split():
                if word not in d:
                    d[word] = index
                    index += 1

    generate_vocab_map(src_vocab_mapping, src_sentences)
    generate_vocab_map(tgt_vocab_mapping, tgt_sentences)

    src_tokens = []
    tgt_tokens = []

    def generate_tokens(tokens_list, sentences, vocab_map):
        """Function to create list of integer tokens"""
        for sentence in sentences:
            sentence_tokens = []
            for word in sentence.split():
                if word in vocab_map:
                    sentence_tokens.append(vocab_map[word])

            tokens_list.append(sentence_tokens)

    generate_tokens(src_tokens, src_sentences, src_vocab_mapping)
    generate_tokens(tgt_tokens, tgt_sentences, tgt_vocab_mapping)


    def generate_padding_tokens(tokens_list, sequence_length):
        """Function to pad the rest of the sentence with zero to maintain the sequence length"""
        padded_tokens_list = []
        for tokens in tokens_list:
            if len(tokens) < sequence_length:
                tokens = tokens + [0] * (sequence_length - len(tokens))
            else:
                tokens = tokens[:sequence_length]
            padded_tokens_list.append(tokens)
        return padded_tokens_list


    def generate_decoder_targets(padded_tokens_list, sequence_length):
        """Function to generate decoder labels one shifted to the right"""
        decoder_targets = []
        for tokens in padded_tokens_list:
            if len(tokens) < sequence_length:
                shifted_tokens = tokens[1:] + [0]
            else:
                shifted_tokens = tokens[1:sequence_length] + [0]
            decoder_targets.append(shifted_tokens)
        return decoder_targets


    padded_src_tokens = generate_padding_tokens(src_tokens, sequence_length)
    padded_tgt_tokens = generate_padding_tokens(tgt_tokens, sequence_length)

    encoder_inputs = np.array(padded_src_tokens)
    decoder_inputs = np.array(padded_tgt_tokens)

    # Decoder labels will be the same targets but each shifted one to the right,
    # Eg: decoder_inputs [0, 1, 2, 3], decoder_targets: [1, 2, 3, 4]
    decoder_targets = np.array(generate_decoder_targets(padded_tgt_tokens, sequence_length))

    return encoder_inputs, decoder_inputs, decoder_targets, src_vocab_mapping, tgt_vocab_mapping

    
from collections import Counter

import torch
import torch.nn as nn
import math

# BertTokenizer class handles tokenization, encoding, and decoding of text
class BertTokenizer:
    def __init__(self, max_len):
        # Maximum length for tokenized sequences
        self.max_len = max_len
        # Initialize vocabulary with special tokens
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        # Vocabulary size counter
        self.vocab_size = len(self.vocab)
        # Inverse vocabulary for decoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    # Builds vocabulary from a list of texts, adding words that appear at least `min_freq` times
    def build_vocab(self, texts, min_freq=2):
        word_counts = Counter()
        # Count occurrences of each word in the texts
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        # Add words to the vocabulary if they meet the minimum frequency and are not already present
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.vocab:
                self.vocab[word] = self.vocab_size
                self.inverse_vocab[self.vocab_size] = word
                self.vocab_size += 1

    # Basic tokenizer that converts text to lowercase and splits by whitespace
    def _tokenize(self, text):
        return text.lower().split()

    # Encodes a single text string into token IDs with optional truncation and padding
    def encode(self, text, truncation=False, padding=False):
        # Tokenize the input text
        tokens = self._tokenize(text)
        # Truncate tokens if required, leaving space for [CLS] and [SEP] tokens
        if truncation:
            tokens = tokens[:self.max_len - 2]

        # Add special tokens [CLS] at the beginning and [SEP] at the end
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # Convert tokens to their corresponding IDs in the vocabulary
        ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(ids)

        # Apply padding if required to reach max length
        if padding:
            pad_length = self.max_len - len(ids)
            ids = ids + [self.vocab["[PAD]"]] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return ids, attention_mask

    # Decodes a list of token IDs back into a text string
    def decode(self, ids):
        # Join the tokens, ignoring [PAD] tokens
        return " ".join([self.inverse_vocab.get(id, "[UNK]") for id in ids if id != self.vocab["[PAD]"]])

    # Enables calling the tokenizer object like a function to encode text
    def __call__(self, texts, truncation=False, padding=False):
        # If input is a single string, encode it
        if isinstance(texts, str):
            return self.encode(texts, truncation, padding)
        # If input is a list of strings, encode each string
        elif isinstance(texts, list):
            return [self.encode(text, truncation, padding) for text in texts]
        else:
            # Raise an error for unsupported input types
            raise ValueError("Input must be a string or a list of strings")

# BertEmbeddings class to handle word and positional embeddings
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout_rate):
        super().__init__()
        # Word embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # Positional embedding layer
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        # Layer normalization to stabilize training
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # Get the sequence length of the input
        seq_length = input_ids.size(1)
        # Create position IDs for each token position
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get word embeddings for input token IDs
        words_embeddings = self.word_embeddings(input_ids)
        # Get positional embeddings for each token position
        position_embeddings = self.position_embeddings(position_ids)

        # Sum word and position embeddings
        embeddings = words_embeddings + position_embeddings
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# BertSelfAttention class implements multi-head self-attention
class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_rate):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        # Size of each attention head
        self.attention_head_size = int(hidden_size / num_attention_heads)
        # Total size for all heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear layers for query, key, and value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    # Helper function to transpose input for multi-head attention
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        # Compute query, key, and value projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize scores to probabilities with softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute context layer as weighted sum of value vectors
        context_layer = torch.matmul(attention_probs, value_layer)
        # Reshape and permute to original shape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

# BertLayer class represents a single transformer block
class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate):
        super().__init__()
        # Self-attention layer
        self.attention = BertSelfAttention(hidden_size, num_attention_heads, dropout_rate)
        # Feed-forward network layers
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        # Layer normalization layers
        self.LayerNorm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Activation function (GELU)
        self.activation = nn.GELU()

    def forward(self, hidden_states, attention_mask=None):
        # Apply self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        # Add residual connection, apply dropout and layer norm
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm1(hidden_states + attention_output)

        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        # Output layer with residual connection, dropout, and layer norm
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm2(attention_output + layer_output)

        return layer_output

# BertModel class inspired from the BERT architecture
class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size, max_position_embeddings, dropout_rate):
        super().__init__()
        # Embedding layer
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, dropout_rate)
        # Encoder: stack of transformer layers
        self.encoder = nn.ModuleList([BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_rate)
                                      for _ in range(num_hidden_layers)])
        # Pooling layer to get a single vector representation
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None):
        # Compute input embeddings
        embedding_output = self.embeddings(input_ids)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Extend attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # Mask out padding tokens
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = embedding_output
        # Pass through each transformer layer
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)

        # Pooling to obtain single vector representation of the sequence
        pooled_output = self.pooler_activation(self.pooler(hidden_states[:, 0]))
        return hidden_states, pooled_output

# BertForSequenceClassification class adds a classification layer on top of BERT
class BertForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size, max_position_embeddings, dropout_rate, num_labels):
        super().__init__()
        # Base BERT model
        self.bert = BertModel(vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                              intermediate_size, max_position_embeddings, dropout_rate)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through BERT model
        _, pooled_output = self.bert(input_ids, attention_mask)
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Compute logits for each class
        logits = self.classifier(pooled_output)
        return logits
# Practical 5: English to French Neural Machine Translation with Attention Mechanism

## Overview

This practical demonstrates the implementation of a neural machine translation system that translates English text to French using an encoder-decoder architecture with Luong attention mechanism. The project is built using TensorFlow/Keras and showcases key concepts in sequence-to-sequence learning and attention mechanisms.

## Table of Contents

- [Key Components](#key-components)
- [Architecture Overview](#architecture-overview)
- [Dataset](#dataset)
- [Model Components](#model-components)
- [Training Process](#training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Performance Analysis](#performance-analysis)
- [Attention Visualization](#attention-visualization)


## Key Components

### **Encoder-Decoder Architecture**
- **Encoder**: Bidirectional LSTM processes English input sequences
- **Decoder**: Unidirectional LSTM generates French output sequences  
- **Attention Mechanism**: Luong attention allows decoder to focus on relevant input parts

### **Core Features**
- Text preprocessing and normalization
- Custom tokenization with vocabulary creation
- Teacher forcing training technique
- BLEU score evaluation
- Attention weight visualization
- Checkpoint saving and loading

## Architecture Overview

### Data Flow
```
English Input → Preprocessing → Tokenization → Encoder → 
Context Vectors → Attention → Decoder → French Output
```

### Model Architecture
1. **Input Processing**: Text cleaning, normalization, tokenization
2. **Encoding**: Bidirectional LSTM processes input sequence
3. **Attention**: Luong mechanism computes context vectors
4. **Decoding**: Unidirectional LSTM generates target sequence
5. **Output**: Probability distribution over French vocabulary

## Dataset

### Source
- **Dataset**: English-French parallel corpus from Anki
- **URL**: http://www.manythings.org/anki/fra-eng.zip
- **Format**: Tab-separated sentence pairs
- **Size**: 30,000 sentence pairs (configurable)

### Data Split
- **Training**: 24,000 samples (80%)
- **Validation**: 6,000 samples (20%)

### Preprocessing Steps
```python
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'\s+', ' ', w)
    w = re.sub(r"[^a-zA-Z?.!,¿<>/0-9]+", ' ', w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w
```

## Model Components

### 1. **LanguageTokenizer Class**
- Creates vocabulary from training corpus
- Handles special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`
- Provides encoding/decoding functionality
- Manages sequence padding and truncation

### 2. **Encoder Class**
```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        # Bidirectional LSTM with state return
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(enc_units,                                                     
                    return_sequences=True, 
                    return_state=True), 
                    merge_mode='sum'
                )
```

### 3. **LuongAttention Class**
```python
class LuongAttention(tf.keras.layers.Layer):
    def call(self, query, values):
        # Compute attention scores and context vector
        values_transformed = self.W(values)
        score = tf.matmul(values_transformed, query_expanded)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
```

### 4. **Decoder Class**
- Unidirectional LSTM with attention integration
- Context-aware prediction generation
- Teacher forcing support during training

### 5. **EncoderDecoderModel Class**
- Complete end-to-end architecture
- Handles both training and inference modes
- Supports different batch sizes for training/inference

## Training Process

### Hyperparameters
```python
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 10
NUM_EXAMPLES = 30000
LEARNING_RATE = 0.001
```

### Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Sparse Categorical Crossentropy with masking
- **Gradient Clipping**: Global norm clipping (max norm: 5.0)
- **Checkpointing**: Every 5 epochs
- **Training Time**: ~32 seconds per epoch (GPU-dependent)

### Loss Function
```python
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    mask = tf.cast(tf.not_equal(real, 0), dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
```

## Evaluation Metrics

### BLEU Score Implementation
- **N-gram Precision**: Calculates 1-gram to 4-gram precision
- **Brevity Penalty**: Penalizes overly short translations
- **Geometric Mean**: Computes final BLEU score

```python
def calculate_bleu_score(reference, candidate, n=4):
    # Calculate precision for each n-gram level
    # Apply brevity penalty
    # Return geometric mean of precisions
```

### Performance Metrics
- **Average BLEU Score**: ~6.05% on validation set
- **Vocabulary Size**: 4,337 English tokens, 7,501 French tokens
- **Max Sequence Length**: English: 9, French: 17

## Results

### Training Progress
```
Epoch 1 Loss: 4.1372 (Time: 81.92 sec)
Epoch 2 Loss: 2.8587 (Time: 32.27 sec)
Epoch 3 Loss: 2.2746 (Time: 32.65 sec)
...
Epoch 10 Loss: 0.5562 (Time: 32.60 sec)
```

### Sample Translations
| English Input | Model Output | Quality |
|---------------|--------------|---------|
| "I love you." | "je t aime ." | Good |
| "How are you?" | "comment vas tu ?" | Good |
| "Good morning." | "bonjour ." | Excellent |
| "Where is the bathroom?" | "ou sont les nouvelles ?" | Poor |

### BLEU Evaluation Results
```
English: <start> i m unwell . <end>
Reference: je ne me sens pas bien .
Predicted: je ne suis pas encore debout .
BLEU Score: 0.00

Average BLEU Score: 6.05
```

## Installation and Setup

### Prerequisites
```bash
pip install tensorflow>=2.8.0
pip install numpy pandas matplotlib scikit-learn
```

### Required Libraries
```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
import time
from sklearn.model_selection import train_test_split
import unicodedata
```

### Dataset Download
```python
def download_and_extract_dataset():
    url = "http://www.manythings.org/anki/fra-eng.zip"
    # Downloads and extracts fra.txt
```

## Usage

### Training the Model
```python
# Run the main training function
if __name__ == '__main__':
    main()
```

### Making Translations
```python
# Translate a single sentence
sentence = "I love you."
result, attention_plot = evaluate_translation(
    sentence, inference_model, en_tokenizer, fr_tokenizer, 
    max_length_en, max_length_fr
)
```

### Visualizing Attention
```python
# Generate attention heatmap
visualize_attention_for_sentence(
    "How are you?", inference_model, en_tokenizer, fr_tokenizer, 
    max_length_en, max_length_fr
)
```

## Implementation Details

### Key Design Decisions

1. **Bidirectional Encoder**: Captures context from both directions
2. **Luong Attention**: Simpler than Bahdanau, effective for this task size
3. **Teacher Forcing**: Accelerates training convergence
4. **Separate Training/Inference Models**: Different batch sizes for efficiency
5. **Gradient Clipping**: Prevents exploding gradients

### Memory Management
- **Sequence Padding**: All sequences padded to maximum length
- **Batch Processing**: Efficient GPU utilization with batch size 64
- **State Management**: Proper LSTM state handling for bidirectional encoder

### Inference Optimization
- **Separate Model Instance**: Batch size 1 for inference
- **Weight Sharing**: Trained weights loaded into inference model
- **Efficient Decoding**: Step-by-step token generation

## Performance Analysis

### Training Characteristics
- **Loss Convergence**: Rapid initial decrease, then gradual improvement
- **Training Time**: First epoch slower due to compilation, then consistent
- **Memory Usage**: Manageable with 64 batch size on standard GPUs

### Model Limitations
1. **Limited Dataset Size**: 30K samples may not capture full language complexity
2. **Simple Architecture**: Basic LSTM may not handle long sequences optimally
3. **Vocabulary Coverage**: OOV words handled with `<unk>` token
4. **Domain Specificity**: Training data bias affects translation quality

## Attention Visualization

### Attention Heatmaps
The implementation generates attention visualizations showing:
- **X-axis**: English input tokens
- **Y-axis**: French output tokens  
- **Color Intensity**: Attention weight strength
- **Alignment Patterns**: How model focuses during translation

### Interpretation
- Diagonal patterns indicate monotonic alignment
- Scattered attention suggests complex linguistic relationships
- Strong attention weights show confident word alignments


## Conclusion

This practical successfully demonstrates the implementation of a neural machine translation system using attention mechanisms. While the BLEU scores indicate room for improvement, the project provides a solid foundation for understanding sequence-to-sequence learning, attention mechanisms, and the challenges involved in neural machine translation.


## References

1. Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate.
3. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks.
4. Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of machine translation.

---

**Course**: DAM202 - Deep Learning and Machine Learning  

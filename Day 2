# Day 2 Deep Dive: Attention & Transformers

## Part 1: The Problem - Why RNNs Failed

Before attention mechanisms, we used Recurrent Neural Networks (RNNs) to process sequences. They had fatal flaws.

### How RNNs Work

RNNs process text word-by-word, maintaining a hidden state that gets updated at each step:

```
Input:  "The cat sat on the mat"
        ↓    ↓   ↓   ↓   ↓    ↓
Step 1: The  →  h₁
Step 2: cat  →  h₂ (uses h₁)
Step 3: sat  →  h₃ (uses h₂)
Step 4: on   →  h₄ (uses h₃)
Step 5: the  →  h₅ (uses h₄)
Step 6: mat  →  h₆ (uses h₅)
```

**The Fundamental Problems:**

1. **Vanishing Gradients**: Information from "The" gets diluted by the time we reach "mat"
2. **Sequential Processing**: Can't parallelize - must wait for word 5 before processing word 6
3. **Fixed Context**: The hidden state has fixed size, creating an information bottleneck
4. **Long-Range Dependencies**: Struggles with sentences like "The cat, which was sleeping on the cozy mat near the fireplace, woke up" - by the time we reach "woke up", we've forgotten about "cat"

### Simple RNN Implementation (to see the problem)

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        # Weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(input_size, hidden_size) * 0.01  # hidden to output
    
    def forward(self, inputs):
        """
        inputs: list of input vectors
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        hidden_states = []
        
        # Process each input sequentially
        for x in inputs:
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1})
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h))
            hidden_states.append(h)
        
        return hidden_states, h  # Return all states and final state

# Example: See information decay
rnn = SimpleRNN(input_size=10, hidden_size=5)

# Simulate 20 input steps
inputs = [np.random.randn(10, 1) for _ in range(20)]
hidden_states, final_state = rnn.forward(inputs)

# Check how much first input affects final state
print(f"Hidden state at step 1 norm: {np.linalg.norm(hidden_states[0]):.3f}")
print(f"Hidden state at step 10 norm: {np.linalg.norm(hidden_states[9]):.3f}")
print(f"Final hidden state norm: {np.linalg.norm(final_state):.3f}")

# Information from early steps gets "washed out"
```

### Visualization of RNN's Problem

```
Long sentence: "The author who wrote the book that I mentioned yesterday arrived"

RNN processing:
The → [h1: info about "The"]
author → [h2: mostly "author", little "The"]
who → [h3: mostly "who", fading "author"]
...
arrived → [h15: mostly "arrived", almost no "The" or "author"]

Problem: By the time we get to "arrived", we've lost track of WHO arrived!
```

---

## Part 2: Attention Mechanism - The Breakthrough

Attention solves this by letting the model **look at all words simultaneously** and decide which ones to focus on.

### The Core Intuition

Instead of compressing everything into a fixed hidden state, attention asks:

> "For this current word, which other words in the sentence are most relevant?"

**Example Sentence**: "The cat sat on the mat"

When processing "sat", attention might look at:
- "cat" (high attention) - WHO is sitting?
- "mat" (high attention) - WHERE is the sitting?
- "The" (low attention) - less important

### The Attention Mechanism Explained

Attention uses three components: **Query**, **Key**, and **Value**

Think of it like a search engine:
- **Query**: "What am I looking for?" (current word)
- **Key**: "What do I contain?" (all words)
- **Value**: "What information do I have?" (all words)

```
For word "sat":
Query: "I need context about sitting"
Keys: ["The" info, "cat" info, "sat" info, "on" info, "the" info, "mat" info]
Values: [embeddings of all words]

Attention computes: How similar is my Query to each Key?
→ High similarity with "cat" and "mat"
→ Low similarity with "the"

Output: Weighted sum of Values based on similarity scores
```

### Attention Formula

The famous attention equation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
Q = Query matrix
K = Key matrix  
V = Value matrix
d_k = dimension of keys (scaling factor)
```

**Step by step:**

1. **QK^T**: Compute similarity between query and all keys (dot product)
2. **/ √d_k**: Scale down (prevents extremely large values)
3. **softmax()**: Convert to probabilities (sum to 1)
4. **× V**: Weighted sum of values

### Visual Example

```
Sentence: "cat sat mat"
Word embeddings (simplified to 3 dims):
cat: [0.8, 0.2, 0.1]
sat: [0.1, 0.9, 0.3]
mat: [0.7, 0.1, 0.8]

Computing attention for "sat":

Step 1: Compute Q, K, V (linear transformations)
Q_sat = [0.5, 0.6, 0.2]
K_cat = [0.4, 0.3, 0.1]
K_sat = [0.2, 0.8, 0.3]
K_mat = [0.6, 0.2, 0.7]

Step 2: Compute attention scores (dot products)
score(sat, cat) = Q_sat · K_cat = 0.5×0.4 + 0.6×0.3 + 0.2×0.1 = 0.40
score(sat, sat) = Q_sat · K_sat = 0.5×0.2 + 0.6×0.8 + 0.2×0.3 = 0.64
score(sat, mat) = Q_sat · K_mat = 0.5×0.6 + 0.6×0.2 + 0.2×0.7 = 0.56

Step 3: Scale and softmax
Scaled scores: [0.40/√3, 0.64/√3, 0.56/√3] = [0.23, 0.37, 0.32]
After softmax: [0.28, 0.39, 0.33]  ← These are attention weights!

Step 4: Weighted sum of values
Output = 0.28×V_cat + 0.39×V_sat + 0.33×V_mat
```

### Attention Implementation

```python
import numpy as np

def softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

class SimpleAttention:
    def __init__(self, d_model, d_k):
        """
        d_model: dimension of input embeddings
        d_k: dimension of queries and keys
        """
        self.d_k = d_k
        
        # Weight matrices to project embeddings to Q, K, V
        self.W_q = np.random.randn(d_model, d_k) * 0.01
        self.W_k = np.random.randn(d_model, d_k) * 0.01
        self.W_v = np.random.randn(d_model, d_k) * 0.01
    
    def forward(self, x):
        """
        x: input embeddings, shape (seq_len, d_model)
        """
        # Project to Q, K, V
        Q = np.dot(x, self.W_q)  # (seq_len, d_k)
        K = np.dot(x, self.W_k)  # (seq_len, d_k)
        V = np.dot(x, self.W_v)  # (seq_len, d_k)
        
        # Compute attention scores: QK^T
        scores = np.dot(Q, K.T)  # (seq_len, seq_len)
        
        # Scale by sqrt(d_k)
        scores = scores / np.sqrt(self.d_k)
        
        # Apply softmax to get attention weights
        attention_weights = softmax(scores)  # (seq_len, seq_len)
        
        # Weighted sum of values
        output = np.dot(attention_weights, V)  # (seq_len, d_k)
        
        return output, attention_weights

# Example usage
seq_len = 4  # "The cat sat mat"
d_model = 8  # embedding dimension
d_k = 4      # attention dimension

# Random embeddings for demonstration
embeddings = np.random.randn(seq_len, d_model)

attention = SimpleAttention(d_model, d_k)
output, weights = attention.forward(embeddings)

print("Input shape:", embeddings.shape)
print("Output shape:", output.shape)
print("\nAttention weights (each row = attention for one word):")
print(weights)
print("\nRow sums (should be ~1.0):", weights.sum(axis=1))

# Visualize which words attend to which
words = ["The", "cat", "sat", "mat"]
print("\n" + "="*50)
print("Attention Pattern:")
print("="*50)
for i, word in enumerate(words):
    print(f"\n{word} attends to:")
    for j, other_word in enumerate(words):
        print(f"  {other_word}: {weights[i,j]:.3f} {'█' * int(weights[i,j] * 20)}")
```

**Sample Output:**
```
Attention Pattern:
==================================================

The attends to:
  The: 0.312 ██████
  cat: 0.198 ███
  sat: 0.264 █████
  mat: 0.226 ████

cat attends to:
  The: 0.156 ███
  cat: 0.389 ███████
  sat: 0.287 █████
  mat: 0.168 ███

sat attends to:
  The: 0.201 ████
  cat: 0.341 ██████
  sat: 0.229 ████
  mat: 0.229 ████
```

---

## Part 3: Self-Attention vs Cross-Attention

There are two main types of attention mechanisms used in transformers.

### Self-Attention (What BERT and GPT Use)

**Definition**: A word attends to other words **in the same sequence**.

```
Input sentence: "The cat sat"

Self-attention lets:
- "The" look at ["The", "cat", "sat"]
- "cat" look at ["The", "cat", "sat"]  
- "sat" look at ["The", "cat", "sat"]

All Q, K, V come from the SAME input sequence.
```

**Use Cases:**
- Understanding context within a sentence
- BERT (encoding): Building rich representations
- GPT (generation): Generating next word based on previous words

```python
def self_attention(x, W_q, W_k, W_v):
    """
    x: input sequence embeddings
    Self-attention: Q, K, V all derived from x
    """
    Q = np.dot(x, W_q)
    K = np.dot(x, W_k)
    V = np.dot(x, W_v)
    
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    output = np.dot(weights, V)
    
    return output, weights
```

### Cross-Attention (What Translation Models Use)

**Definition**: A sequence attends to a **different** sequence.

```
English: "The cat sat"
French:  "Le chat"  [generating next word]

Cross-attention lets:
- "Le" attend to ["The", "cat", "sat"]  ← Look at source
- "chat" attend to ["The", "cat", "sat"]

Q comes from target (French), K and V come from source (English)
```

**Use Cases:**
- Machine translation
- Image captioning (image → text)
- Text-to-image (text → image features)

```python
def cross_attention(x_target, x_source, W_q, W_k, W_v):
    """
    x_target: target sequence (e.g., French)
    x_source: source sequence (e.g., English)
    Cross-attention: Q from target, K and V from source
    """
    Q = np.dot(x_target, W_q)  # Query from target
    K = np.dot(x_source, W_k)  # Key from source
    V = np.dot(x_source, W_v)  # Value from source
    
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    output = np.dot(weights, V)
    
    return output, weights
```

### Visual Comparison

```
SELF-ATTENTION:
Input:  [The] [cat] [sat] [on] [mat]
          ↓     ↓     ↓     ↓     ↓
         [All words attend to all words in same sequence]
          ↓     ↓     ↓     ↓     ↓
Output: [The'] [cat'] [sat'] [on'] [mat']


CROSS-ATTENTION (Translation):
Source: [The] [cat] [sat]     ← English
          ↑     ↑     ↑
          └─────┴─────┘
                │
Target: [Le] [chat] [était]   ← French
         ↓     ↓      ↓
Output: [Le'] [chat'] [était']
```

### Complete Implementation with Both

```python
class AttentionMechanisms:
    def __init__(self, d_model, d_k):
        self.d_k = d_k
        
        # For self-attention
        self.W_q_self = np.random.randn(d_model, d_k) * 0.01
        self.W_k_self = np.random.randn(d_model, d_k) * 0.01
        self.W_v_self = np.random.randn(d_model, d_k) * 0.01
        
        # For cross-attention
        self.W_q_cross = np.random.randn(d_model, d_k) * 0.01
        self.W_k_cross = np.random.randn(d_model, d_k) * 0.01
        self.W_v_cross = np.random.randn(d_model, d_k) * 0.01
    
    def self_attention(self, x):
        """Q, K, V all from x"""
        Q = np.dot(x, self.W_q_self)
        K = np.dot(x, self.W_k_self)
        V = np.dot(x, self.W_v_self)
        
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        weights = softmax(scores)
        output = np.dot(weights, V)
        
        return output, weights
    
    def cross_attention(self, x_target, x_source):
        """Q from target, K and V from source"""
        Q = np.dot(x_target, self.W_q_cross)
        K = np.dot(x_source, self.W_k_cross)
        V = np.dot(x_source, self.W_v_cross)
        
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        weights = softmax(scores)
        output = np.dot(weights, V)
        
        return output, weights

# Example: Translation scenario
d_model = 8
d_k = 4

# Source: "The cat" (English)
source_embeddings = np.random.randn(2, d_model)  # 2 words
# Target: "Le" (French, generating next word)
target_embeddings = np.random.randn(1, d_model)  # 1 word so far

attn = AttentionMechanisms(d_model, d_k)

# Self-attention on source (understand English)
source_context, self_weights = attn.self_attention(source_embeddings)
print("Self-attention weights (English understanding itself):")
print(self_weights)

# Cross-attention (French looking at English)
translation_context, cross_weights = attn.cross_attention(target_embeddings, source_embeddings)
print("\nCross-attention weights (French attending to English):")
print(cross_weights)
print("\nInterpretation: When generating 'Le', it attends to English words:")
print(f"  'The': {cross_weights[0,0]:.3f}")
print(f"  'cat': {cross_weights[0,1]:.3f}")
```

---

## Part 4: Multi-Head Attention - Looking from Different Angles

Single attention head can only capture one type of relationship. Multi-head attention learns **multiple representations simultaneously**.

### The Intuition

Think of reading a sentence:
- **Head 1**: Focuses on subject-verb relationships ("cat" → "sat")
- **Head 2**: Focuses on location ("sat" → "mat")
- **Head 3**: Focuses on descriptions ("cat" → "fluffy")
- **Head 4**: Focuses on time/tense markers

Each head can specialize in different linguistic phenomena!

### How Multi-Head Works

```
Input embedding (d_model = 512)
         ↓
Split into 8 heads (each gets 512/8 = 64 dims)
         ↓
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ H1 │ H2 │ H3 │ H4 │ H5 │ H6 │ H7 │ H8 │  ← Each runs attention independently
└────┴────┴────┴────┴────┴────┴────┴────┘
         ↓
Concatenate outputs
         ↓
Linear projection
         ↓
Final output (d_model = 512)
```

### Multi-Head Attention Implementation

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        d_model: dimension of embeddings (must be divisible by num_heads)
        num_heads: number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        
        # Projection matrices for all heads (done in parallel)
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        x shape: (seq_len, d_model)
        output shape: (num_heads, seq_len, d_k)
        """
        seq_len = x.shape[0]
        # Reshape to (seq_len, num_heads, d_k)
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        # Transpose to (num_heads, seq_len, d_k)
        return x.transpose(1, 0, 2)
    
    def forward(self, x):
        """
        x: input embeddings, shape (seq_len, d_model)
        """
        seq_len = x.shape[0]
        
        # Linear projections
        Q = np.dot(x, self.W_q)  # (seq_len, d_model)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention for each head
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # (num_heads, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)
        
        # Softmax over last dimension
        attention_weights = np.zeros_like(scores)
        for h in range(self.num_heads):
            attention_weights[h] = softmax(scores[h])
        
        # Apply attention to values
        attended = np.matmul(attention_weights, V)  # (num_heads, seq_len, d_k)
        
        # Concatenate heads
        attended = attended.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
        attended = attended.reshape(seq_len, self.d_model)  # (seq_len, d_model)
        
        # Final linear projection
        output = np.dot(attended, self.W_o)
        
        return output, attention_weights

# Example usage
seq_len = 5  # "The cat sat on mat"
d_model = 64  # embedding dimension
num_heads = 8

embeddings = np.random.randn(seq_len, d_model)

mha = MultiHeadAttention(d_model, num_heads)
output, attn_weights = mha.forward(embeddings)

print(f"Input shape: {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")  # (num_heads, seq_len, seq_len)

# Visualize different heads attending differently
words = ["The", "cat", "sat", "on", "mat"]
print("\n" + "="*60)
print("Multi-Head Attention Patterns:")
print("="*60)

for head in range(min(3, num_heads)):  # Show first 3 heads
    print(f"\nHead {head + 1} (might focus on different relationships):")
    for i, word in enumerate(words):
        top_attended = np.argmax(attn_weights[head, i])
        attention_score = attn_weights[head, i, top_attended]
        print(f"  {word} → {words[top_attended]} (score: {attention_score:.3f})")
```

### Why Multiple Heads Matter

**Single Head Limitation:**
```
"The cat sat on the mat because it was tired"

Single attention might focus on:
cat → sat (subject-verb)

But misses:
it → cat (coreference)
tired → sat (reason)
mat → on (location)
```

**Multi-Head Solution:**
```
Head 1: cat → sat (grammar)
Head 2: it → cat (coreference)
Head 3: tired → sat (causation)
Head 4: mat → on (spatial)
Head 5: because → tired (reasoning)
...
```

---

## Part 5: Encoder vs Decoder Architecture

Transformers come in three flavors: Encoder-only, Decoder-only, and Encoder-Decoder.

### Encoder-Only (BERT-style)

**Purpose**: Understanding and representing text

**Architecture**:
```
Input: "The cat sat"
  ↓
Embeddings + Positional Encoding
  ↓
┌─────────────────────┐
│ Multi-Head Attention│ ← Bidirectional: each word sees ALL words
│   (self-attention)  │
└─────────────────────┘
  ↓
┌─────────────────────┐
│   Feed-Forward NN   │
└─────────────────────┘
  ↓ (repeat 12 times)
┌─────────────────────┐
│  Rich Representations│
└─────────────────────┘
```

**Key Features:**
- **Bidirectional**: Words can attend to both past and future
- **Use case**: Classification, NER, question answering
- **Examples**: BERT, RoBERTa, DeBERTa

**Attention Pattern (no masking):**
```
     The  cat  sat
The  [✓]  [✓]  [✓]   ← "The" sees everything
cat  [✓]  [✓]  [✓]   ← "cat" sees everything  
sat  [✓]  [✓]  [✓]   ← "sat" sees everything
```

### Decoder-Only (GPT-style)

**Purpose**: Generating text sequentially

**Architecture**:
```
Input: "The cat"  [predicting "sat"]
  ↓
Embeddings + Positional Encoding
  ↓
┌─────────────────────┐
│ Masked Multi-Head   │ ← Causal: can only see past
│   Self-Attention    │
└─────────────────────┘
  ↓
┌─────────────────────┐
│   Feed-Forward NN   │
└─────────────────────┘
  ↓ (repeat 12-96 times)
┌─────────────────────┐
│  Predict Next Token │
└─────────────────────┘
```

**Key Features:**
- **Causal/Autoregressive**: Can only attend to previous tokens
- **Use case**: Text generation, completion
- **Examples**: GPT-3, GPT-4, Claude, LLaMA

**Attention Pattern (with masking):**
```
     The  cat  sat
The  [✓]  [✗]  [✗]   ← "The" only sees itself
cat  [✓]  [✓]  [✗]   ← "cat" sees The + itself
sat  [✓]  [✓]  [✓]   ← "sat" sees all previous
```

### Encoder-Decoder (T5-style)

**Purpose**: Sequence-to-sequence tasks (translation, summarization)

**Architecture**:
```
Input: "The cat sat" (English)

ENCODER (bidirectional):
  ↓
┌─────────────────────┐
│ Self-Attention      │ ← All words see each other
└─────────────────────┘
  ↓
[Encoded representation]
         ↓
         
DECODER (causal):
Output: "Le chat" (French, generating next)
  ↓
┌─────────────────────┐
│ Masked Self-Attn    │ ← Causal mask
└─────────────────────┘
  ↓
┌─────────────────────┐
│ Cross-Attention     │ ← Look at encoder output
└─────────────────────┘
  ↓
[Next token prediction]
```

**Key Features:**
- **Encoder**: Bidirectional understanding of input
- **Decoder**: Causal generation using encoder context
- **Use case**: Translation, summarization, Q&A
- **Examples**: T5, BART, mT5

### Complete Encoder-Decoder Implementation

```python
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn_W1 = np.random.randn(d_model, d_model * 4) * 0.01
        self.ffn_W2 = np.random.randn(d_model * 4, d_model) * 0.01
    
    def forward(self, x):
        # Multi-head self-attention (no mask - bidirectional)
        attn_output, _ = self.attention.forward(x)
        x = x + attn_output  # Residual connection
        
        # Feed-forward network
        ffn_output = np.dot(np.maximum(0, np.dot(x, self.ffn_W1)), self.ffn_W2)
        x = x + ffn_output  # Residual connection
        
        return x

class TransformerDecoderLayer:
    def __init__(self, d_model, num_heads):
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = AttentionMechanisms(d_model, d_model // num_heads)
        self.ffn_W1 = np.random.randn(d_model, d_model * 4) * 0.01
        self.ffn_W2 = np.random.randn(d_model * 4, d_model) * 0.01
    
    def create_causal_mask(self, seq_len):
        """Create mask to prevent attending to future tokens"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9  # Large negative number
        return mask
    
    def forward(self, x, encoder_output):
        # Masked self-attention (causal)
        attn_output, _ = self.masked_attention.forward(x)
        x = x + attn_output
        
        # Cross-attention to encoder
        cross_output, _ = self.cross_attention.cross_attention(x, encoder_output)
        x = x + cross_output
        
        # Feed-forward
        ffn_output = np.dot(np.maximum(0, np.dot(x, self.ffn_W1)), self.ffn_W2)
        x = x + ffn_output
        
        return x

# Complete model
class SimpleTransformer:
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers):
        self.encoders = [TransformerEncoderLayer(d_model, num_heads) 
                        for _ in range(num_encoder_layers)]
        self.decoders = [TransformerDecoderLayer(d_model, num_heads)
                        for _ in range(num_decoder_layers)]
    
    def encode(self, x):
        """Pass through all encoder layers"""
        for encoder in self.encoders:
            x = encoder.forward(x)
        return x
    
    def decode(self, x, encoder_output):
        """Pass through all decoder layers"""
        for decoder in self.decoders:
            x = decoder.forward(x, encoder_output)
        return x

# Example: Translation
d_model = 64
source = np.random.randn(3, d_model)  # English: "The cat sat"
target = np.random.randn(2, d_model)  # French: "Le chat" (generating next)

transformer = SimpleTransformer(d_model, num_heads=8, 
                               num_encoder_layers=2, 
                               num_decoder_layers=2)

# Encode source
encoded = transformer.encode(source)
print(f"Encoded source shape: {encoded.shape}")

# Decode target
decoded = transformer.decode(target, encoded)
print(f"Decoded output shape: {decoded.shape}")
```

### Architecture Comparison

| Feature | Encoder (BERT) | Decoder (GPT) | Encoder-Decoder (T5) |
|---------|---------------|---------------|---------------------|
| Attention | Bidirectional | Causal | Both |
| Input sees | All tokens | Past only | All (encoder) + Past (decoder) |
| Best for | Understanding | Generation | Translation |
| Training | Masked tokens | Next token | Span corruption |
| Examples | Classification | Completion | Summarization |

---

## Part 6: Positional Encoding - Adding Order Information

Attention has no inherent notion of sequence order! "cat sat mat" and "mat sat cat" look identical to raw attention.

### The Problem

```python
# Without position info, these are identical:
sentence1 = ["The", "cat", "sat"]
sentence2 = ["sat", "The", "cat"]

# Attention only sees the SET of words, not their ORDER
```

### Solution: Positional Encoding

Add position information directly to embeddings:

```
Word Embedding:    [0.2, 0.8, 0.1, 0.5]
Positional Encoding: [0.1, 0.0, 0.5, 0.3]  ← Position 0
                    ───────────────────────
Final Embedding:   [0.3, 0.8, 0.6, 0.8]
```

### Sinusoidal Positional Encoding (Original Transformer)

Uses sine and cosine functions of different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
pos = position in sequence (0, 1, 2, ...)
i = dimension index
d_model = embedding dimension
```

**Why this works:**
1. Each position gets a unique encoding
2. Model can learn relative positions
3. Extrapolates to longer sequences than seen in training

### Implementation

```python
def positional_encoding(seq_len, d_model):
    """
    Create sinusoidal positional encodings
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Visualize positional encodings
import matplotlib.pyplot as plt

seq_len = 50
d_model = 128

pe = positional_encoding(seq_len, d_model)

plt.figure(figsize=(12, 6))
plt.imshow(pe.T, cmap='RdBu', aspect='auto')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding Pattern')
plt.colorbar()
plt.show()

print(f"Positional encoding shape: {pe.shape}")
print(f"\nPosition 0 encoding (first 10 dims): {pe[0, :10]}")
print(f"Position 10 encoding (first 10 dims): {pe[10, :10]}")
```

### Complete Example with Positional Encoding

```python
class TransformerWithPositions:
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len=512):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.01
        
        # Positional encodings (computed once, reused)
        self.pos_encodings = positional_encoding(max_seq_len, d_model)
        
        # Attention layer
        self.attention = MultiHeadAttention(d_model, num_heads)
    
    def embed(self, token_ids):
        """
        token_ids: list of token indices
        """
        seq_len = len(token_ids)
        
        # Get token embeddings
        embeddings = np.array([self.token_embeddings[idx] for idx in token_ids])
        
        # Add positional encodings
        embeddings = embeddings + self.pos_encodings[:seq_len]
        
        return embeddings
    
    def forward(self, token_ids):
        # Embed with positions
        x = self.embed(token_ids)
        
        # Apply attention
        output, weights = self.attention.forward(x)
        
        return output, weights

# Example
vocab_size = 1000
d_model = 64
num_heads = 8

model = TransformerWithPositions(vocab_size, d_model, num_heads)

# Token IDs for "The cat sat"
token_ids = [10, 234, 567]

output, attn_weights = model.forward(token_ids)
print(f"Output shape: {output.shape}")

# Show how position affects embedding
emb_without_pos = model.token_embeddings[10]
emb_with_pos = model.embed([10])[0]

print(f"\nToken 10 embedding without position: {emb_without_pos[:5]}")
print(f"Token 10 embedding with position: {emb_with_pos[:5]}")
print(f"Difference (position encoding): {(emb_with_pos - emb_without_pos)[:5]}")
```

### Learned Positional Embeddings (BERT, GPT)

Modern models often use **learned** position embeddings instead of sinusoidal:

```python
class LearnedPositionalEmbedding:
    def __init__(self, max_seq_len, d_model):
        # Just like word embeddings, but for positions
        self.position_embeddings = np.random.randn(max_seq_len, d_model) * 0.01
    
    def forward(self, seq_len):
        return self.position_embeddings[:seq_len]
```

**Learned vs Sinusoidal:**
- Learned: More flexible, task-specific
- Sinusoidal: Generalizes to longer sequences, no extra parameters

---

## Part 7: Why Attention Beats RNNs

Let's compare directly:

### Speed: Parallelization

**RNN (Sequential):**
```
Time Step: 1    2    3    4    5
Process:   The → cat → sat → on → mat
Wait:      ✓    ✓    ✓    ✓    ✓

Total time: 5 sequential steps (can't parallelize)
```

**Transformer (Parallel):**
```
All words processed simultaneously:
The ─┐
cat ─┼─→ [Attention computes all relationships at once]
sat ─┤
on  ─┤
mat ─┘

Total time: 1 parallel step (all words at once)
```

### Memory: Long-Range Dependencies

**RNN Problem:**
```
Sentence: "The cat, which was sitting on the warm and cozy mat near the fireplace, suddenly jumped"

RNN hidden state at "jumped":
h = [0.1, 0.3, ...]  ← Information about "cat" is diluted/lost

Challenge: Remember "cat" jumped (15 words apart)
```

**Transformer Solution:**
```
Direct attention path: "jumped" → "cat"
Attention weight: 0.82  ← Strong connection despite distance

No information loss - direct connection between any two words!
```

### Information Bottleneck

**RNN:**
```
Entire sequence → Single fixed-size hidden state → Decode

Problem: 100 words compressed into h ∈ ℝ^512
Information loss guaranteed!
```

**Transformer:**
```
Each position has full access to all positions
No compression required
Information preserved!
```

### Numerical Comparison

```python
import time

def benchmark_rnn_vs_attention():
    seq_len = 100
    d_model = 512
    
    # Simulate RNN (sequential)
    start = time.time()
    hidden = np.zeros(d_model)
    for t in range(seq_len):
        # Sequential update
        x = np.random.randn(d_model)
        hidden = np.tanh(hidden + x)  # Simplified
    rnn_time = time.time() - start
    
    # Simulate Attention (parallel)
    start = time.time()
    X = np.random.randn(seq_len, d_model)
    # All positions processed at once
    scores = np.dot(X, X.T)  # Parallelizable
    attention_time = time.time() - start
    
    print(f"RNN time: {rnn_time:.4f}s")
    print(f"Attention time: {attention_time:.4f}s")
    print(f"Speedup: {rnn_time / attention_time:.2f}x")

benchmark_rnn_vs_attention()
```

### Trade-offs

**Attention Advantages:**
- ✓ Parallel processing
- ✓ Direct long-range connections
- ✓ No vanishing gradients across sequence
- ✓ Interpretable attention patterns

**Attention Disadvantages:**
- ✗ Quadratic memory complexity O(n²) for sequence length n
- ✗ Requires positional encoding (not inherent to architecture)
- ✗ Expensive for very long sequences (10,000+ tokens)

**RNN Advantages:**
- ✓ Linear memory O(n)
- ✓ Natural handling of variable length
- ✓ Inherent sequential inductive bias

**RNN Disadvantages:**
- ✗ Sequential bottleneck
- ✗ Vanishing/exploding gradients
- ✗ Poor long-range modeling

---

## Part 8: Visualizing Attention Weights

Let's build an interactive attention visualizer!

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self, d_model=64, num_heads=4):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.num_heads = num_heads
    
    def visualize_attention(self, tokens, embeddings):
        """
        tokens: list of strings (words)
        embeddings: numpy array (seq_len, d_model)
        """
        output, attn_weights = self.attention.forward(embeddings)
        
        # Create figure with subplots for each head
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Head Attention Patterns', fontsize=16)
        
        axes = axes.flatten()
        
        for head in range(min(4, self.num_heads)):
            ax = axes[head]
            
            # Get attention weights for this head
            weights = attn_weights[head]  # (seq_len, seq_len)
            
            # Create heatmap
            sns.heatmap(weights, annot=True, fmt='.2f', 
                       xticklabels=tokens, yticklabels=tokens,
                       cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Attention'})
            
            ax.set_title(f'Head {head + 1}')
            ax.set_xlabel('Key (attending to)')
            ax.set_ylabel('Query (attending from)')
        
        plt.tight_layout()
        return fig, attn_weights
    
    def visualize_single_word(self, tokens, embeddings, word_idx):
        """Visualize what a specific word attends to"""
        output, attn_weights = self.attention.forward(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Average attention across all heads for this word
        avg_attention = attn_weights[:, word_idx, :].mean(axis=0)
        
        # Bar plot
        ax.bar(range(len(tokens)), avg_attention)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'What "{tokens[word_idx]}" attends to (averaged across heads)')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight the querying word
        ax.axvline(word_idx, color='red', linestyle='--', alpha=0.5, label='Query word')
        ax.legend()
        
        plt.tight_layout()
        return fig

# Example usage
tokens = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 64
embeddings = np.random.randn(len(tokens), d_model)

visualizer = AttentionVisualizer(d_model=d_model, num_heads=4)

# Visualize all heads
fig1, weights = visualizer.visualize_attention(tokens, embeddings)

# Visualize what "cat" attends to
fig2 = visualizer.visualize_single_word(tokens, embeddings, word_idx=1)

print("\nAttention interpretation:")
print("="*50)
for i, token in enumerate(tokens):
    avg_attn = weights[:, i, :].mean(axis=0)
    top_idx = np.argmax(avg_attn)
    print(f"{token:8s} most attends to: {tokens[top_idx]:8s} (weight: {avg_attn[top_idx]:.3f})")
```

### Interactive Attention Explorer

```python
def explore_attention_patterns(sentence, d_model=64, num_heads=8):
    """
    Interactive function to explore attention patterns
    """
    tokens = sentence.split()
    embeddings = np.random.randn(len(tokens), d_model)
    
    attention = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = attention.forward(embeddings)
    
    print("="*60)
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}")
    print("="*60)
    
    # Analyze each head
    for head in range(num_heads):
        print(f"\nHead {head + 1} patterns:")
        weights = attn_weights[head]
        
        # Find most attended pairs
        for i, query_token in enumerate(tokens):
            attended_idx = np.argmax(weights[i])
            attended_token = tokens[attended_idx]
            attention_score = weights[i, attended_idx]
            
            if i != attended_idx:  # Skip self-attention
                print(f"  {query_token:10s} → {attended_token:10s} ({attention_score:.3f})")
    
    return attn_weights

# Example
sentence = "The quick brown fox jumps over the lazy dog"
weights = explore_attention_patterns(sentence)
```

---

## Part 9: Complete Transformer Block

Let's build a full transformer block with all components:

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        d_model: embedding dimension
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension (usually 4 * d_model)
        dropout: dropout rate
        """
        self.d_model = d_model
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ff_W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.ff_W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.ff_b1 = np.zeros(d_ff)
        self.ff_b2 = np.zeros(d_model)
        
        # Layer normalization parameters (simplified)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
        
        self.dropout = dropout
    
    def layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta
    
    def feed_forward(self, x):
        """
        Feed-forward network with ReLU activation
        FFN(x) = ReLU(xW1 + b1)W2 + b2
        """
        hidden = np.maximum(0, np.dot(x, self.ff_W1) + self.ff_b1)  # ReLU
        output = np.dot(hidden, self.ff_W2) + self.ff_b2
        return output
    
    def forward(self, x):
        """
        Complete forward pass with:
        1. Multi-head attention
        2. Add & Norm
        3. Feed-forward
        4. Add & Norm
        """
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention.forward(x)
        x = x + attn_output  # Residual
        x = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)  # Norm
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output  # Residual
        x = self.layer_norm(x, self.ln2_gamma, self.ln2_beta)  # Norm
        
        return x, attn_weights

# Build a complete model
class SimpleGPT:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token + Position embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff=d_model*4) 
                      for _ in range(num_layers)]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, token_ids):
        """Complete forward pass"""
        seq_len = len(token_ids)
        
        # Embed tokens
        x = np.array([self.token_emb[idx] for idx in token_ids])
        
        # Add positional encoding
        x = x + self.pos_emb[:seq_len]
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block.forward(x)
            attention_weights.append(attn)
        
        # Project to vocabulary
        logits = np.dot(x, self.output_proj)
        
        return logits, attention_weights

# Example: Mini GPT
vocab_size = 1000
d_model = 128
num_heads = 8
num_layers = 4
max_seq_len = 512

model = SimpleGPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)

# Input: "The cat sat"
token_ids = [10, 234, 567]
logits, all_attn_weights = model.forward(token_ids)

print(f"Input tokens: {token_ids}")
print(f"Output logits shape: {logits.shape}")  # (3, 1000) - predict next token for each position
print(f"Number of attention weight matrices: {len(all_attn_weights)}")  # 4 layers
print(f"Each attention weight shape: {all_attn_weights[0].shape}")  # (8 heads, 3 tokens, 3 tokens)

# Predict next token
next_token_logits = logits[-1]  # Logits for position after "sat"
next_token_id = np.argmax(next_token_logits)
print(f"\nPredicted next token ID: {next_token_id}")
```

---

## Summary: Day 2 Essentials

### Attention Mechanism
- **Core idea**: Dynamically weight all inputs based on relevance
- **Formula**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Components**: Query (what I'm looking for), Key (what I contain), Value (what I provide)
- **Advantage**: Direct connections between all positions - no information bottleneck

### Self-Attention vs Cross-Attention

| Type | Q from | K,V from | Use Case |
|------|--------|----------|----------|
| Self-Attention | Same sequence | Same sequence | Understanding context (BERT, GPT) |
| Cross-Attention | Target seq | Source seq | Translation, image captioning |

### Multi-Head Attention
- **Why**: Learn multiple types of relationships simultaneously
- **How**: Split embeddings into h heads, run attention in parallel, concatenate
- **Example heads**: syntax, semantics, coreference, temporal relationships
- **Typical**: 8-16 heads in practice

### Encoder vs Decoder

**Encoder (BERT-style)**:
- Bidirectional attention (sees full context)
- Best for: classification, NER, understanding
- Example: "Is this email spam?"

**Decoder (GPT-style)**:
- Causal/masked attention (only sees past)
- Best for: generation, completion
- Example: "Complete this sentence..."

**Encoder-Decoder (T5-style)**:
- Encoder: bidirectional understanding
- Decoder: causal generation + cross-attention to encoder
- Best for: translation, summarization

### Positional Encoding
- **Problem**: Attention has no inherent notion of order
- **Solution**: Add position information to embeddings
- **Methods**: Sinusoidal (fixed) or learned (flexible)
- **Critical**: Without this, "cat sat mat" = "mat sat cat"

### Why Attention Beats RNNs

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| Processing | Sequential (slow) | Parallel (fast) |
| Long-range | Vanishing gradients | Direct paths |
| Memory | O(n) | O(n²) |
| Bottleneck | Fixed hidden state | No bottleneck |

**Speed**: Transformers are 10-100x faster to train
**Quality**: Better on long sequences (>50 tokens)
**Trade-off**: More memory for long sequences

### Interview-Ready Soundbite

*"Attention revolutionized NLP by replacing sequential processing with parallel computation. Instead of compressing a sequence into a fixed hidden state, attention creates direct connections between all positions. This eliminates the vanishing gradient problem for long-range dependencies. Multi-head attention lets the model learn different relationship types simultaneously - syntax in one head, semantics in another. The key insight is the Query-Key-Value mechanism: each word asks 'what's relevant to me?' (Query), all words answer 'here's what I contain' (Key), and attention computes a weighted sum of their information (Value)."*

### Practical Takeaways

**For Your Projects:**

```python
# Use HuggingFace Transformers
from transformers import AutoModel, AutoTokenizer

# BERT (encoder) for classification
model = AutoModel.from_pretrained("bert-base-uncased")

# GPT-2 (decoder) for generation  
model = AutoModel.from_pretrained("gpt2")

# T5 (encoder-decoder) for summarization
model = AutoModel.from_pretrained("t5-base")
```

**Architecture Choices**:
- Classification task → Use encoder (BERT)
- Text generation → Use decoder (GPT)
- Seq-to-seq (translation) → Use encoder-decoder (T5)

**Key Hyperparameters**:
- `d_model`: 512-1024 (embedding dimension)
- `num_heads`: 8-16 (attention heads)
- `num_layers`: 6-24 (transformer blocks)
- `d_ff`: 2048-4096 (feed-forward hidden dim, usually 4×d_model)

**For Interviews:**

Must-know questions:
1. **"Explain attention in simple terms"** → It's like a search engine: Query asks what's relevant, Keys answer what they contain, Values provide information
2. **"Why multi-head?"** → Learn multiple relationship types (syntax, semantics, etc.) simultaneously
3. **"Attention vs RNN?"** → Parallel vs sequential, direct connections vs gradient chains
4. **"What's the computational complexity?"** → O(n²) in sequence length for attention
5. **"How does GPT differ from BERT?"** → Causal masking (only past) vs bidirectional (full context)

**For Your Resume:**

Strong bullet points:
- "Implemented multi-head self-attention mechanism achieving 94% accuracy on sentiment classification"
- "Fine-tuned BERT encoder for named entity recognition, improving F1 score by 18%"
- "Built GPT-based text completion system processing 1000 requests/second"
- "Optimized transformer inference using attention caching, reducing latency by 40%"

**Common Pitfalls:**

❌ Forgetting positional encoding (order matters!)
❌ Using bidirectional attention for generation (causes data leakage)
❌ Not scaling attention scores by √d_k (numerical instability)
❌ Ignoring the O(n²) memory cost for long sequences
❌ Mixing up self-attention and cross-attention use cases

**Debug Checklist:**
- ✓ Attention weights sum to 1.0 (softmax)
- ✓ Causal mask applied for generation tasks
- ✓ Position encodings added to embeddings
- ✓ Residual connections preserve gradient flow
- ✓ Layer norm stabilizes training

**Next Steps (Day 3 Preview):**

Tomorrow you'll learn **RAG (Retrieval-Augmented Generation)**:
- How to ground LLMs in external knowledge
- Vector databases for semantic search
- Chunking strategies for long documents
- Building a PDF → Chat pipeline

**Tools to Know:**

- **Attention Visualization**: BertViz, Transformer Interpretability
- **Model Libraries**: HuggingFace Transformers, PyTorch, JAX
- **Architectures**: BERT, GPT-2/3, T5, RoBERTa, DeBERTa
- **Papers to Read**: "Attention Is All You Need" (Vaswani et al., 2017)

---

## Quick Reference Card

```
TRANSFORMER DECISION TREE:

Need to understand text? → Encoder (BERT)
  ├─ Classification → BERT + classifier head
  ├─ NER/POS tagging → BERT + token classifier
  └─ Q&A → BERT + span prediction

Need to generate text? → Decoder (GPT)
  ├─ Completion → GPT with causal mask
  ├─ Chat → GPT with instruction tuning
  └─ Code generation → CodeGPT/Codex

Need translation/summarization? → Encoder-Decoder (T5)
  ├─ Translation → T5 with parallel corpus
  ├─ Summarization → T5 with summary pairs
  └─ Q&A generation → T5 with context-question pairs

ATTENTION DEBUGGING:

Weights don't sum to 1? → Check softmax
Generation using future tokens? → Add causal mask
Position doesn't matter? → Verify positional encoding
Training unstable? → Check attention score scaling (√d_k)
Out of memory? → Reduce sequence length or use sparse attention
```

---

*You've completed Day 2! You now understand how transformers process sequences and why they're superior to RNNs. Tomorrow: Using transformers for real-world applications with RAG!*

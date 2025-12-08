# Install: pip install transformers torch sentence-transformers

# ============================================
# OPTION 1: HuggingFace Tokenizers (Most Common)
# ============================================
from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained tokenizer (BPE-based)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization and embeddings are fundamental to LLMs!"

# Tokenize
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# Output: ['token', '##ization', 'and', 'em', '##bed', '##ding', '##s', ...]

# Get token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [101, 19204, 3989, 1998, 7861, 8270, 2015, ...]

# Decode back
decoded = tokenizer.decode(token_ids)
print("Decoded:", decoded)

# ============================================
# OPTION 2: Get Embeddings (BERT)
# ============================================
model = AutoModel.from_pretrained("bert-base-uncased")

# Prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # Shape: [1, seq_len, 768]

print(f"Embedding shape: {embeddings.shape}")
print(f"First token embedding (first 5 dims): {embeddings[0][0][:5]}")

# Get sentence embedding (mean pooling)
sentence_embedding = embeddings.mean(dim=1)
print(f"Sentence embedding shape: {sentence_embedding.shape}")

# ============================================
# OPTION 3: Sentence Transformers (Easiest!)
# ============================================
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embeddings directly
sentences = [
    "I love natural language processing",
    "NLP is amazing",
    "I enjoy cooking pasta"
]

embeddings = model.encode(sentences)
print(f"\nEmbedding shape: {embeddings.shape}")  # (3, 384)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)
print("\nSimilarity matrix:")
print(similarities)
# Sentence 1 & 2 will have high similarity (~0.7-0.8)
# Sentence 1 & 3 will have low similarity (~0.1-0.3)

# ============================================
# OPTION 4: OpenAI/Cohere (Production)
# ============================================
# pip install openai
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here"
)

embedding = response.data[0].embedding
print(f"OpenAI embedding length: {len(embedding)}")  # 1536 dimensions

# ============================================
# QUICK COMPARISON
# ============================================
"""
Library            | Use Case                    | Embedding Dims
-------------------|-----------------------------|-----------------
BERT               | General NLP tasks           | 768
Sentence-BERT      | Semantic search, similarity | 384-1024
GPT-2 Tokenizer    | Text generation            | 50257 vocab
OpenAI Embeddings  | Production apps            | 1536
Word2Vec (gensim)  | Legacy/custom corpus       | 100-300
"""

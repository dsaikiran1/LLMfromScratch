import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type=str, required=True, help="Please provide a batch_size")

args = parser.parse_args()

print(f"batch_size:{args.batch_size}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
block_size = 16
batch_size = 32
max_iters = 200
learning_rate = 3e-3
eval_iters=100
n_embd = 200
n_head = 4
dropout = 0.2
n_layer = 4

chars=""
with open('openwebtext/vocab.txt','r',encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
vocab_size = len(chars)
print(chars)

string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])


def get_random_chunk(split):
    filename "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f: "openwebtext/wal_split.txt"
        with mmap.mmap(f.fileno(), 0, access-mmap.ACCESS_READ) as mm:
        # Determine the file size and a random position to start reading
            file_size=len(mm)
            start_pos=random.randint(0, (file_size) block_size*batch size)
            #Seek to the random position and read the block of text 
            mm.seek(start_pos)
            block=mm.read(block_size*batch_size-1)
            #Decode the block to a string, ignoring any invalid byte sequences 
            decoded_block=block.decode('utf-8', errors='ignore').replace('\r', '')
            #Train and test splits
            data=torch.tensor(encode(decoded_block), dtype torch.long)
    return data

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Check for shape consistency
    x_list = [data[i:i+block_size] for i in ix]
    y_list = [data[i+1:i+block_size+1] for i in ix]
    
    assert all(x.shape[0] == block_size for x in x_list), "Mismatch in block size"
    assert all(y.shape[0] == block_size for y in y_list), "Mismatch in block size"
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # Print device assignment for x and y
    #print(f"x device: {x.device}, y device: {y.device}")

    x, y = x.to(device), y.to(device)
    return x, y

# Check the batch creation process
x, y = get_batch('train')
print('inputs:')
print(x)
print('targets:')
print(y)

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape  # Batch size, Sequence length, Embedding size

        # Calculate Key, Query, Value matrices
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        # Self-attention scores (scaled dot-product attention)
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)
        # Apply mask to ensure causal structure (look only at past tokens)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Softmax to get attention weights
        wei = F.softmax(wei, dim=-1)
        # Dropout on the attention weights (for regularization)
        wei = self.dropout(wei)
        # Weighted sum of value vectors
        out = wei @ v  # (B, T, head_size)
        return out
        
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel  """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #(B,T,F) -> (B,T ,[h1,h1,h1,h1,h2,h2,h2,h2,h3,h3,h3,h3])
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 + n_embd),
            nn.ReLU(),
            nn.Linear(4 + n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x
         
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _init_weights(self):
        """Custom weight initialization for GPT model"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for Linear layers
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Zero-initialize biases
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Normal initialization for embeddings
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)  # Initialize weight for LayerNorm
                nn.init.zeros_(module.bias)   # Initialize bias for LayerNorm

    def forward(self, index, targets):
        B,T = index.shape
        tok_emb = self.token_embedding_table(index) #(B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))#(B, T, C)
        x = tok_emb + pos_emb#(B, T, C)
        x = self.blocks(x)#(B, T, C)
        x = self.ln_f(x)#(B, T, C)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits,loss
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index,None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index,index_next),dim=1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
m = model.to(device)
with open('model-01.pkl','rb') as f:
    model = pickle.load()
print("model loaded succefully")
n = model.to(device)

model = GPTLanguageModel(vocab_size)  # Initialize your model
model = model.to(device)  # Move the model to the correct device (CPU or GPU)

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)


with open('model-01.pkl','wb') as f:
    pickle.dump(model, f)
print('model saved')

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0),max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
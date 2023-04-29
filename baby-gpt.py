import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters -----------------------------------------------
block_size = 64 # what is the maximum context length for predictions.
batch_size = 64 # how many independent character sequences should we process in parallel.
learning_rate = 1e-3
device = "cpu"
# device = 'mps' if torch.mps.is_available() else 'cpu'
max_iters = 5000 # number of training iterations
eval_interval = 100
eval_iters = 200
n_embd = 128
dropout = 0.2
# ----------------------------------------------------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

itos = {i:ch for i,ch in enumerate(chars)}
stoi = {ch:i for i,ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s] # encoder: take a string, output a list of integers.
decode = lambda l: ''.join([itos[num] for num in l]) # decoder: take a list of integers, output a string.

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix], dim=0)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix], dim=0)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C=head_size)
        q = self.query(x) # (B,T,C=head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C=head_size) @ (B, C=head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0.0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        self.dropout(wei)
        
        # perform the weighted aggregation of values
        v = self.value(x) # (B,T,C=head_size)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C=head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # n_embd: Embedding dimensions, n_head: the number of heads we'd like
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # creating an embedding lookup table.
        # each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (B, T) -> (B, T, C) where C = n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (B, T) -> (B, T, C)

        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers.
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # position embedding in the column will be broadcasted across B rows.
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets != None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits , loss
        else:
            return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # crop ids to the last block size tokens.
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            # focus only on the last time step.
            logits = logits[:, -1, :]
            # apply softmax to get prediction probabilities.
            probs = F.softmax(logits, dim=1) # B, C
            # sample from the probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
  
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):

  # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, 1000)[0].tolist()))


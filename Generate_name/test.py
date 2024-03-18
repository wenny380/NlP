import torch
import torch.nn.functional as F

def build_dataset(my_list, stoi1, block_s=3):
    X, Y = [], []
    for w in my_list:
        context = [0] * block_s
        for ch in w + '.':
            ix = stoi1[ch]
            X.append(context) # array X store the current running context
            Y.append(ix)  # array of the current character
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    #print(X.shape, Y.shape)
    return X, Y

#read file
words = open("test.txt", 'r', encoding='utf-8').read().splitlines()
#put all the letter in lowercase
words = [x.lower() for x in words]

#build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
Count_chars = len(itos)


#build dataset
block_size = 3 # context length
Xte, Yte = build_dataset(words, stoi, block_size)
#load model
model = torch.load("model.torch")
C, W1, b1, W2, b2 = model['C'], model['W1'], model['b1'], model['W2'], model['b2']


emb = C[Xte] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
#Calculate the loss for the training set
loss = F.cross_entropy(logits, Yte)
print("Loss test: ", loss)

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        emb = C[torch.tensor([context])] # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

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


words = open("train.txt", 'r', encoding='utf-8').read().splitlines()
words = [x.lower() for x in words]


# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
Count_chars = len(itos)
#print(itos)


# build the dataset
block_size = 3 # context length
Xtr, Ytr = build_dataset(words, stoi, block_size)
#print(Xtr.shape, Ytr.shape)

#initialize the C tensor matrix
C = torch.randn((Count_chars, 2))
ys = C[:, 1]
xs = C[:, 0]
tmp=torch.arange(6).view(-1, 3)
#embedding
emb = C[Xtr]
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
#change the shape of our embedding variable
emb.view(-1, 6).shape

#the hidden layer
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

W2 = torch.randn((100, Count_chars)) #weights
b2 = torch.randn(Count_chars) #bias
logits = h @ W2 + b2

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((Count_chars, 10), generator=g)
W1 = torch.randn((30, 200), generator=g) #200 neurons
b1 = torch.randn(200, generator=g) #200 bias
W2 = torch.randn((200, Count_chars), generator=g) #200 input
b2 = torch.randn(Count_chars, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

for i in range(20000):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]] # (32, 3, 10)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
    logits = h @ W2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad


emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
#Calculate the loss for the training set
loss = F.cross_entropy(logits, Ytr)
print("Loss: ", loss)
#loss: 1.67

#save the modele
modele_name = 'model.torch'
torch.save({'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, modele_name)
print("the modele for name generation is saved")

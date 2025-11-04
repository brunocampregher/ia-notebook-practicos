"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 50 # number of steps to unroll the RNN for
learning_rate = 1e-2

# model parameters
#Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
#Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
Wxz = np.random.randn(hidden_size, vocab_size)*0.01 # input a actualizacion
Whz = np.random.randn(hidden_size, hidden_size)*0.01 # hidden a actualizacion
Wxn = np.random.randn(hidden_size, vocab_size)*0.01 
Whn = np.random.randn(hidden_size, hidden_size)*0.01
Wxr = np.random.randn(hidden_size, vocab_size)*0.01
Whr = np.random.randn(hidden_size, hidden_size)*0.01
br = np.zeros((hidden_size, 1))
bn = np.zeros((hidden_size, 1))
bz = np.zeros((hidden_size, 1)) # bias actualizacion
#bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, zs, rs, ys, ps, hcs = {}, {}, {}, {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):

    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    zs[t] = sigmoid(np.dot(Wxz, xs[t]) + np.dot(Whz, hs[t - 1]) + bz)
    rs[t] = sigmoid(np.dot(Wxr, xs[t]) + np.dot(Whr, hs[t - 1] + br))
    hcs[t] = np.tanh(np.dot(Wxn, xs[t]) + rs[t] * np.dot(Whn, hs[t - 1]) + bn) # hidden candidate state
    hs[t] = (1 - zs[t]) * hs[t - 1] + zs[t] * hcs[t] # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxz, dWhz, dbz = np.zeros_like(Wxz), np.zeros_like(Whz), np.zeros_like(bz)
  # (r) Reset gate
  dWxr, dWhr, dbr = np.zeros_like(Wxr), np.zeros_like(Whr), np.zeros_like(br)
  # (n) Candidate state
  dWxn, dWhn, dbn = np.zeros_like(Wxn), np.zeros_like(Whn), np.zeros_like(bn)
  dWhy, dby = np.zeros_like(Why), np.zeros_like(by)

  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhc = dh * zs[t]
    dz = dh * (hcs[t] - hs[t - 1])
    dh_prev_h = dh * (1 - zs[t])
    #dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dzraw = (zs[t] * (1 - zs[t])) * dz
    dhcraw = (1 - hcs[t]**2) * dhc

    #dbh += dhraw
    dbn += dhcraw
    dWxn += np.dot(dhcraw, xs[t].T)

    dr = dhcraw * np.dot(Whn, hs[t - 1])
    dh_prev_hcs = np.dot(Whn.T, dhcraw * rs[t])
    dWhn += np.dot(dhcraw * rs[t], hs[t - 1].T)

    dbz += dzraw
    dWxz += np.dot(dzraw, xs[t].T)

    dh_prev_z = np.dot(Whz.T, dzraw)
    dWhz += np.dot(dzraw, hs[t - 1].T)

    drraw = (rs[t] * (1 - rs[t])) * dr
    dbr += drraw
    dWxr += np.dot(drraw, xs[t].T)

    dh_prev_r = np.dot(Whr.T, drraw) 
    dWhr += np.dot(drraw, hs[t - 1].T)

    dhnext = dh_prev_h + dh_prev_hcs + dh_prev_z + dh_prev_r
  for dparam in [dWxz, dWhz, dbz, dWxr, dWhr, dbr, dWxn, dWhn, dbn, dWhy, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxz, dWhz, dbz, dWxr, dWhr, dbr, dWxn, dWhn, dbn, dWhy, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    r = sigmoid(np.dot(Wxr, x) + np.dot(Whr, h) + br) # Cuánto del estado anterior se olvida
    z = sigmoid(np.dot(Wxz, x) + np.dot(Whz, h) + bz) # Cuánto del estado anterior se mantiene
    h_can = np.tanh(np.dot(Wxn, x) + r * np.dot(Whn, h) + bn)
    h = (1 - z) * h + z * h_can
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxz, mWhz, mbz = np.zeros_like(Wxz), np.zeros_like(Whz), np.zeros_like(bz)
# (r) Reset gate
mWxr, mWhr, mbr = np.zeros_like(Wxr), np.zeros_like(Whr), np.zeros_like(br)
# (n) Candidate state
mWxn, mWhn, mbn = np.zeros_like(Wxn), np.zeros_like(Whn), np.zeros_like(bn)

# (Output)
mWhy, mby = np.zeros_like(Why), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxz, dWhz, dbz, dWxr, dWhr, dbr, dWxn, dWhn, dbn, dWhy, dby, hprev = lossFun(inputs, targets, hprev)  
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxz, Whz, bz,
                Wxr, Whr, br,
                Wxn, Whn, bn,
                Why, by], 
                                [dWxz, dWhz, dbz,
                 dWxr, dWhr, dbr,
                 dWxn, dWhn, dbn,
                 dWhy, dby], 
                                [mWxz, mWhz, mbz,
              mWxr, mWhr, mbr,
              mWxn, mWhn, mbn,
              mWhy, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 

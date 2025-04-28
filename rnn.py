import numpy as np

# 1. Define your training data
text_sequences = [
    ["this", "is", "my", "dog"]
]

# 2. Create a vocabulary and mapping
vocabulary = set()
for seq in text_sequences:
    vocabulary.update(seq)
vocabulary = sorted(list(vocabulary))
word_to_index = {word: i for i, word in enumerate(vocabulary)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocabulary)

# 3. Prepare the data for the RNN
def prepare_data(sequences, word_to_index):
    X = []
    y = []
    for seq in sequences:
        input_seq = [word_to_index[word] for word in seq[:3]]
        target_word = word_to_index[seq[3]]
        X.append(input_seq)
        y.append(target_word)
    return np.array(X), np.array(y)

X_train, y_train = prepare_data(text_sequences, word_to_index)

# 4. Define the RNN parameters
hidden_size = 5  # Reduced hidden size for a small dataset
learning_rate = 0.1
epochs = 5000  # Increased epochs to try and memorize

# Initialize weights (randomly)
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # Input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # Hidden to output
bh = np.zeros((hidden_size, 1))  # Hidden bias
by = np.zeros((vocab_size, 1))  # Output bias

# 5. Define the RNN forward and backward pass
def rnn_step_forward(x, h_prev, Wxh, Whh, bh):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    return h

def softmax(vector):
    e = np.exp(vector - np.max(vector))
    return e / e.sum(axis=0)

def loss_function(logits, target):
    probs = softmax(logits)
    loss = -np.log(probs[target, 0])
    return loss, probs

def rnn_forward(inputs, targets, h_prev, Wxh, Whh, Why, bh, by):
    loss = 0
    hidden_states = []
    y_hats = []
    for t in range(len(inputs)):
        x = np.zeros((vocab_size, 1))
        x[inputs[t], 0] = 1  # One-hot encode the input word
        h_prev = rnn_step_forward(x, h_prev, Wxh, Whh, bh)
        hidden_states.append(h_prev)
        logits = np.dot(Why, h_prev) + by
        loss_t, probs = loss_function(logits, targets)  # Use 'targets' directly
        loss += loss_t
        y_hats.append(probs)
    return loss / len(inputs), y_hats, hidden_states

def rnn_step_backward(dy_next, dh_next, dWhy, dby, dh, h_prev, h_current, x, Wxh, Whh):
    dtanh = (1 - h_current ** 2) * dh
    dWxh = np.dot(dtanh, x.T)
    dWhh = np.dot(dtanh, h_prev.T)
    dbh = dtanh
    dx = np.dot(Wxh.T, dtanh)
    dh_prev = np.dot(Whh.T, dtanh)
    return dWxh, dWhh, dbh, dx, dh_prev

def rnn_backward(inputs, targets, hidden_states, y_hats, Wxh, Whh, Why, bh, by):
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hidden_states[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(y_hats[t])
        dy[targets, 0] -= 1  # Derivative of the cross-entropy loss - use 'targets' directly
        dWhy += np.dot(dy, hidden_states[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dh_next
        x = np.zeros((vocab_size, 1))
        x[inputs[t], 0] = 1
        h_prev = hidden_states[t-1] if t > 0 else np.zeros_like(hidden_states[0])
        dWxh_t, dWhh_t, dbh_t, dx, dh_prev = rnn_step_backward(None, dh, dWhy, dby, dh, h_prev, hidden_states[t], x, Wxh, Whh)
        dWxh += dWxh_t
        dWhh += dWhh_t
        dbh += dbh_t
        dh_next = dh_prev
    return dWxh, dWhh, dWhy, dbh, dby

# 6. Training loop
for epoch in range(epochs):
    total_loss = 0
    h_prev = np.zeros((hidden_size, 1))  # Initialize hidden state
    for i in range(len(X_train)):
        inputs = X_train[i]
        target = y_train[i]

        loss, y_hats, hidden_states = rnn_forward(inputs, target, h_prev, Wxh, Whh, Why, bh, by) # Pass 'target' directly
        total_loss += loss

        dWxh, dWhh, dWhy, dbh, dby = rnn_backward(inputs, target, hidden_states, y_hats, Wxh, Whh, Why, bh, by) # Pass 'target' directly

        # Gradient clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update parameters
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(X_train):.4f}")

# 7. Prediction function
def predict_next_word(input_sequence, word_to_index, index_to_word, Wxh, Whh, Why, bh, by, h_prev):
    input_indices = [word_to_index[word] for word in input_sequence]
    for t in range(len(input_indices)):
        x = np.zeros((len(word_to_index), 1))
        x[input_indices[t], 0] = 1
        h_prev = rnn_step_forward(x, h_prev, Wxh, Whh, bh)
    logits = np.dot(Why, h_prev) + by
    probs = softmax(logits)
    predicted_index = np.argmax(probs)
    return index_to_word[predicted_index]

# Example prediction
test_sequence = ["this", "is", "my"]
initial_hidden_state = np.zeros((hidden_size, 1))
predicted_word = predict_next_word(test_sequence, word_to_index, index_to_word, Wxh, Whh, Why, bh, by, initial_hidden_state)
print(f"For the sequence '{test_sequence}', the predicted next word is: {predicted_word}")
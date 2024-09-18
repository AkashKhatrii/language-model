from scripts.utils import get_files, convert_files2idx, convert_line2idx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import math
import pickle
import torch.utils.data as data
from tqdm import tqdm

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.manual_seed(42)

with open ( './data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

idx_2_char = {}
def idx2char():
    for char in vocab:
        idx_2_char[vocab[char]] = char

idx2char()
# print(idx_2_char)


class CharRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers = 1):
        super().__init__()
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        
        self.embedding = nn.Embedding(386, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_dim, 386)
    
    def forward(self, x, h, c):
        # BATCH SIZE = 64
        out = self.embedding(x) # In: (64, 500) Out: (64, 500, 50) 
        out, (h, c) = self.lstm(out, (h, c)) # In: (64, 500, 50) Out: (64, 500, 200)
        lstm_output = torch.reshape(out,(out.shape[0]*out.shape[1],out.shape[2])) # In: (64, 500, 200) Out: (64 X 500, 200) = (32000, 200)
        fc1_output = self.fc1(lstm_output) # In: (32000, 200) Out: (32000, 200)
        relu_output = self.relu(fc1_output) # Out: (32000, 200)
        fc2_output = self.fc2(relu_output) # In: (32000, 200) Out: (32000, 386)
        out = torch.reshape(fc2_output,(out.shape[0], out.shape[1], fc2_output.shape[1])) # OUT: (64, 500, 386)
    
        return out, h, c
    

model_path =  r"C:\Users\akash\OneDrive\Desktop\Akash\UofU\Coursework\Fall_2023\NLPwithDL\mp3\best_model_2"
checkpoint = torch.load(model_path)

best_num_layer = checkpoint["num_layers"]
best_lr = checkpoint["lr"]

model = CharRNN(embedding_dim=50, hidden_dim=200, num_layers=best_num_layer)
model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=best_lr)
# criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(lw).to(device).to(torch.float32), ignore_index=384, reduce=False)

model.load_state_dict(checkpoint["model_param"])
# optimizer.load_state_dict(checkpoint["optim_param"])

model.eval()

seed_sequence = "The little boy was" 
generated_sequence = seed_sequence

hidden = torch.zeros(2, 1, 200).to(device) 
context = torch.zeros(2, 1, 200).to(device)

seed_int = [[vocab[char] for char in seed_sequence]]

with torch.no_grad():
    seed_sequence_input = torch.tensor(seed_int).to(device)  # Convert the sequence to a tensor
    print(seed_sequence_input.shape)
    print(len(seed_sequence_input))
    _, hidden, context = model(seed_sequence_input, hidden, context)

last_char = generated_sequence[-1]
# print(idx_2_char)
for _ in range(200):
    last_char_int = [[vocab[last_char]]]
    # print(last_char_int)
    last_char_input = torch.tensor(last_char_int).to(device) 
    output, hidden, context = model(last_char_input, hidden, context)
    # print(output.shape)
    # output = output.squeeze(0)
    # print(output)
    next_char_index = torch.argmax(output).item()
    # print(next_char_index)
    next_char = idx_2_char[next_char_index]
    last_char = next_char
    generated_sequence += next_char

# Print the generated sequence
print(generated_sequence)


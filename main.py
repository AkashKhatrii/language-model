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
# print(vocab)

train_files = get_files(r'data\train')
# train_files = train_files[:5]
dev_files = get_files(r'data\dev')
# dev_files = dev_files[:5]
# test_files = get_files(r'data\test')
# test_files = test_files[:1]


class CharRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, seq_length = 500, num_layers = 1):
        super().__init__()
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        
        self.embedding = nn.Embedding(386, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_dim, 386)
    
    def forward(self, x):
        # BATCH SIZE = 64
        # In and Out dimensions written for my understanding.
        out = self.embedding(x) # In: (64, 500) Out: (64, 500, 50) 
        out, (_, _) = self.lstm(out) # In: (64, 500, 50) Out: (64, 500, 200)
        lstm_output = torch.reshape(out,(out.shape[0]*out.shape[1],out.shape[2])) # In: (64, 500, 200) Out: (64 X 500, 200) = (32000, 200)
        fc1_output = self.fc1(lstm_output) # In: (32000, 200) Out: (32000, 200)
        relu_output = self.relu(fc1_output) # Out: (32000, 200)
        fc2_output = self.fc2(relu_output) # In: (32000, 200) Out: (32000, 386)
        out = torch.reshape(fc2_output,(out.shape[0], out.shape[1], fc2_output.shape[1])) # OUT: (64, 500, 386)

        # out = self.embedding(x)
        # lstm_output, (_, _) = self.lstm(out)
        # fc1_out = self.fc1(lstm_output)
        # relu_out = self.relu(fc1_out)
        # fc2_out = self.fc2(relu_out)
    
        return out




# Characters
train_data_X = []
train_data_y = []
dev_data_X = []
dev_data_y = []
test_data_X = []
test_data_y = []


# Characters to indices
train_data_X_idx = []
train_data_y_idx = []
dev_data_X_idx = []
dev_data_y_idx = []
test_data_X_idx = []
test_data_y_idx = []

char_counts = {i:0 for i in range(len(vocab))}

def create_train_data(path, X_list, y_list):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        
        paragraph = ""
        for line in data:
            paragraph += line
        
        numberOfSubSeq = int(len(paragraph) / 500)
        numberOfRemainingChars = len(paragraph) % 500
        numberOfPadTokens = 500 - numberOfRemainingChars

        for i in range(numberOfSubSeq):
            x = paragraph[500 * i : 500 * (i + 1)]

            y = x[1:]
            y += paragraph[500*(i + 1)]

            X_list.append(list(x))
            y_list.append(list(y))


        if numberOfRemainingChars:
            last_entry_x = list(paragraph[-numberOfRemainingChars:])
            last_entry_x.extend(['[PAD]'] * numberOfPadTokens)

            last_entry_y = last_entry_x[1:]
            last_entry_y.append("[PAD]")

            X_list.append(last_entry_x)
            y_list.append(last_entry_y)

def convert2idx(char_arr, idx_arr ):
    for data in char_arr:
        modified_data = []
        for char in data:
            if char not in vocab.keys():
                modified_data.append(vocab["<unk>"])
                char_counts[vocab["<unk>"]] += 1
            else:
                modified_data.append(vocab[char])
                char_counts[vocab[char]] += 1
        idx_arr.append(modified_data)

    # for data in train_data_y:
    #     modified_data = []
    #     for char in data:
    #         if char not in vocab.keys():
    #             modified_data.append(vocab["<unk>"])
    #         else:
    #             modified_data.append(vocab[char])
    #     train_data_y_idx.append(modified_data)




# Creating training data and converting to indices

# Uncomment this

for file in train_files:
    create_train_data(file, train_data_X, train_data_y)
for file in dev_files:
    create_train_data(file, dev_data_X, dev_data_y)
# for file in test_files:
#     create_train_data(file, test_data_X, test_data_y)


print(len(train_data_X), len(train_data_y))
print(len(test_data_X), len(test_data_y))
print(len(dev_data_X), len(dev_data_y))


convert2idx(train_data_X, train_data_X_idx)
convert2idx(train_data_y, train_data_y_idx)
convert2idx(dev_data_X, dev_data_X_idx)
convert2idx(dev_data_y, dev_data_y_idx)
convert2idx(test_data_X, test_data_X_idx)
convert2idx(test_data_y, test_data_y_idx)
# print(train_data_X_idx[0])


# Uncomment this

X = torch.tensor(train_data_X_idx, dtype=torch.float32).reshape(len(train_data_X), 500)
y = torch.tensor(train_data_y_idx)
X_dev = torch.tensor(dev_data_X_idx, dtype=torch.float32).reshape(len(dev_data_X), 500)
y_dev = torch.tensor(dev_data_y_idx)
# X_test = torch.tensor(test_data_X_idx, dtype=torch.float32).reshape(len(test_data_X), 500)
# y_test = torch.tensor(test_data_y_idx)

# print(X.shape, y.shape)

# Uncomment this

train_dataLoader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=64)
dev_dataLoader = data.DataLoader(data.TensorDataset(X_dev, y_dev), shuffle=True, batch_size=64)
# test_dataLoader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=True, batch_size=64)
# creating weight array for loss function
char_count = np.array(list(char_counts.values()))
char_sum = np.sum(char_count,keepdims=True)
char_count = char_count / char_sum
lw = 1 - char_count


# Creating model
# model = CharRNN(embedding_dim=50, hidden_dim=200, seq_length=500, num_layers=2)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(lw).to(device).to(torch.float32), ignore_index=384, reduce=False)
# optimzer = optim.Adam(model.parameters(), lr = 0.0001)


# Hyper Parameters

# Uncomment this

lr = [0.0001, 0.00001, 0.000001]
num_layers = [2]
best_dev_loss = sys.maxsize
best_epoch = -1
best_lr = 0
best_num_layers = 0
best_perplexity = sys.maxsize
num_parameters = 0


# Uncomment this

for layer in num_layers:
    # [1, 2]
    print()
    print("-"*20 + "Number of Layers: " + str(layer) + "-"*20)
    for learning_rate in lr:
        # [0.0001, 0.00001, 0.000001]
        print()
        print("-"*20 + "Learning Rate: " + str(learning_rate) + "-"*20)
        learning_rate_loss = []
        learning_rate_perplexity = sys.maxsize


        model = CharRNN(embedding_dim=50, hidden_dim=200, seq_length=500, num_layers=layer)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(lw).to(device).to(torch.float32), ignore_index=384, reduce=False)
        optimzer = optim.Adam(model.parameters(), lr = learning_rate)
        
        for epoch in range(5):
            with open(r'results.txt', 'a') as f:
                f.write(f"\nEpoch: {epoch + 1}")
            epoch_loss = []
            perplexities = []
            for X, y in tqdm(train_dataLoader):
                X = X.to(device).to(torch.int64)
                y = y.to(device).to(torch.int64)


                model.train()
                model.zero_grad()

                out = model(X)
            
                loss = criterion(out.reshape(out.shape[0], out.shape[2], out.shape[1]), y)
                mask = (y != 384).float()
                non_pad_loss = (loss * mask)
                non_pad_loss_sum = non_pad_loss.sum()

                individual_losses = [loss for loss in non_pad_loss]
                mean_individual_losses = [loss.mean() for loss in individual_losses]
                p = [(math.e)**mean_loss for mean_loss in mean_individual_losses]
                perplexities.extend(p)

                no_of_non_pad_tokens = mask.sum()

                mean_loss = non_pad_loss_sum / no_of_non_pad_tokens
                loss = mean_loss
            
                epoch_loss.append(loss.cpu().item())

                loss.backward()
                optimzer.step()

            model.eval()

            dev_epoch_loss = []
            dev_perplexities = []

            with torch.no_grad():
                for X_dev, y_dev in tqdm(dev_dataLoader):
                    X_dev= X_dev.to(device).to(torch.int64)
                    y_dev = y_dev.to(device).to(torch.int64)

                    out = model(X_dev)
                
                    loss = criterion(out.reshape(out.shape[0], out.shape[2], out.shape[1]), y_dev)
                    mask = (y_dev != 384).float()
                    non_pad_loss = (loss * mask)
                    non_pad_loss_sum = non_pad_loss.sum()

                    individual_dev_losses = [loss for loss in non_pad_loss]
                    mean_dev_individual_losses = [loss.mean() for loss in individual_dev_losses]
                    p = [(math.e)**mean_loss for mean_loss in mean_dev_individual_losses]
                    dev_perplexities.extend(p)

                    no_of_non_pad_tokens = mask.sum()

                    mean_loss = non_pad_loss_sum / no_of_non_pad_tokens
                    loss = mean_loss
                
                    dev_epoch_loss.append(loss.cpu().item())


            learning_rate_loss.extend(dev_epoch_loss) # adding the epoch loss to learning rate loss

            
            print(f"Average training loss: {np.mean(epoch_loss)}")
            print(f"Average Dev Loss: {np.mean(dev_epoch_loss)}")

            average_train_perplexity = sum(perplexities) / len(perplexities)
            print(f"Epoch {epoch + 1} train perplexity: ", average_train_perplexity.item())

            average_dev_perplexity = sum(dev_perplexities) / len(dev_perplexities)
            print(f"Epoch {epoch + 1} dev perplexity: ", average_dev_perplexity.item())

            if learning_rate_perplexity > average_dev_perplexity:
                learning_rate_perplexity = average_dev_perplexity

        if best_perplexity > learning_rate_perplexity:
            best_perplexity = learning_rate_perplexity
            best_lr = learning_rate
            best_num_layers = layer
            num_parameters = sum(p.numel() for p in model.parameters())
        
            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimzer.state_dict(),
                "dev_perplexity": best_perplexity, 
                "num_layers": best_num_layers,
                "lr": best_lr,
                "num_parameters": num_parameters
            }, f"best_model_22")
            
            # Don't uncomment

            # with open(r'results.txt', 'a') as f:
            #     f.write(f"\nLoss: {np.mean(epoch_loss)}")
            #     f.write(f"\nPerplexity: {average_train_perplexity}")

    
# Uncomment this

print(f"Best perplexity: {best_perplexity}")
print("Best lr: {:.6f}".format(best_lr))
print(f"Best num layers: {best_num_layers}")



# model_path =  r"C:\Users\akash\OneDrive\Desktop\Akash\UofU\Coursework\Fall_2023\NLPwithDL\mp3\best_model_1"
# checkpoint = torch.load(model_path)

# best_num_layer = checkpoint["num_layers"]
# best_lr = checkpoint["lr"]

# model = CharRNN(embedding_dim=50, hidden_dim=200, seq_length=500, num_layers=best_num_layer)
# model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=best_lr)
# criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(lw).to(device).to(torch.float32), ignore_index=384, reduce=False)


# model.load_state_dict(checkpoint["model_param"])
# optimizer.load_state_dict(checkpoint["optim_param"])

# model.eval()


# test_epoch_loss = []
# test_perplexities = []

# with torch.no_grad():
#     for X_test, y_test in tqdm(test_dataLoader):
#         X_test = X_test.to(device).to(torch.int64)
#         y_test = y_test.to(device).to(torch.int64)

#         out = model(X_test)
                
#         loss = criterion(out.reshape(out.shape[0], out.shape[2], out.shape[1]), y_test)
#         mask = (y_test != 384).float()
#         non_pad_loss = (loss * mask)
#         non_pad_loss_sum = non_pad_loss.sum()

#         individual_test_losses = [loss for loss in non_pad_loss]
#         mean_test_individual_losses = [loss.mean() for loss in individual_test_losses]
#         p = [(math.e)**mean_loss for mean_loss in mean_test_individual_losses]
#         test_perplexities.extend(p)

#         no_of_non_pad_tokens = mask.sum()

#         mean_loss = non_pad_loss_sum / no_of_non_pad_tokens
#         loss = mean_loss
                
#         test_epoch_loss.append(loss.cpu().item())



# average_test_perplexity = sum(test_perplexities) / len(test_perplexities)
# print(f"Test perplexity: ", average_test_perplexity.item())


# for X, y in train_dataLoader:
#     print(X.shape)
#     print(y.shape)











 
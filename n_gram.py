from scripts.utils import get_files, convert_files2idx, convert_line2idx
from numba import jit, cuda
import math
import pickle
with open ( './data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
# print(vocab)

train_files = get_files(r'data\train')
test_files = get_files(r'data\test')

four_grams_count = {}
trigrams_count = {}

def create_ngram_data(path):
    paragraph = ""
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()

        for line in data:

            if len(line) == 0:
                continue
            elif len(line) == 1:
                four_gram = ['[PAD]', '[PAD]', '[PAD]', line[0]]
                trigram = four_gram[:3]
                four_grams_count[tuple(four_gram)] = 1 + four_grams_count.get(tuple(four_gram), 0)
                trigrams_count[tuple(trigram)] = 1 + trigrams_count.get(tuple(trigram), 0)

            elif len(line) == 2:
                for i in range(2):
                    four_gram = []
                    trigram = []
                    for j in range(3 - i):
                        four_gram.append("[PAD]")
                    for k in range(i + 1):
                        four_gram.append(line[k])
                    trigram = four_gram[:3]
                    four_grams_count[tuple(four_gram)] = 1 + four_grams_count.get(tuple(four_gram), 0)
                    trigrams_count[tuple(trigram)] = 1 + trigrams_count.get(tuple(trigram), 0)
            else:
                for i in range(3):
                    four_gram = []
                    trigram = []
                    for j in range(3 - i):
                        four_gram.append("[PAD]")
                    for k in range(i + 1):
                        four_gram.append(line[k])
                    trigram = four_gram[:3]

                    four_grams_count[tuple(four_gram)] = 1 + four_grams_count.get(tuple(four_gram), 0)
                    trigrams_count[tuple(trigram)] = 1 + trigrams_count.get(tuple(trigram), 0)

            for i in range(len(line) - 3):
                four_gram = [line[i], line[i + 1], line[i + 2], line[i + 3]]
                trigram = four_gram[:3]
                four_grams_count[tuple(four_gram)] = 1 + four_grams_count.get(tuple(four_gram), 0)
                trigrams_count[tuple(trigram)] = 1 + trigrams_count.get(tuple(trigram), 0)

   

    # four gram with maximum count
    # print(max(four_grams_count, key = lambda x: four_grams_count[x])  )
    
test_four_gram = {}
global_loss = 0
global_paragraph_len = 0
global_sentences = 0
global_perplexity = 0
loss_count = 0

def calculate_probability(four_gram, vocab):

    loss = 0
    numerator = four_grams_count.get((four_gram[0], four_gram[1], four_gram[2], four_gram[3]), 0) + 1
    denominator = trigrams_count.get((four_gram[0], four_gram[1], four_gram[2]), 0) + len(vocab)

    probability = (numerator) / (denominator)
    loss = -math.log(probability)

    return loss

def test_ngram(path):
    global global_loss, global_paragraph_len, global_sentences, global_perplexity, loss_count
    paragraph = ""
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        global_sentences += len(data)
        for line in data:
            # paragraph += line
            loss = 0
            if len(line) == 0:
                continue
            elif len(line) == 1:
                four_gram = ['[PAD]', '[PAD]', '[PAD]', line[0]]
                f_loss = calculate_probability(four_gram, vocab)
                loss += f_loss
                loss_count += 1

            elif len(line) == 2:
                for i in range(2):
                    four_gram = []
                    trigram = []
                    for j in range(3 - i):
                        four_gram.append("[PAD]")
                    for k in range(i + 1):
                        four_gram.append(line[k])

                    f_loss = calculate_probability(four_gram, vocab)
                    loss += f_loss
                    loss_count += 1
            else:
                for i in range(3):
                    four_gram = []
                    trigram = []
                    for j in range(3 - i):
                        four_gram.append("[PAD]")
                    for k in range(i + 1):
                        four_gram.append(line[k])

                    f_loss = calculate_probability(four_gram, vocab)
                    loss += f_loss
                    loss_count += 1


            for i in range(len(line) - 3):
                four_gram = [line[i], line[i + 1], line[i + 2], line[i + 3]]
                f_loss = calculate_probability(four_gram, vocab)
                loss += f_loss
                loss_count += 1
            
            perplexity = (math.e) ** (loss/len(line))
            # print("Sentence perplexity: ", perplexity)
            global_perplexity += perplexity
            global_loss += (loss / (len(line)))
        

train_data = []

for file in train_files:
    create_ngram_data(file)
for file in test_files:
    test_ngram(file)

# test_ngram(r'mytest.txt')
perplexity = global_perplexity /global_sentences
print("Perplexity: ", perplexity)
print(len(four_grams_count))


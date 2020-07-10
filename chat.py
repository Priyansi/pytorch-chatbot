import random
import json
import torch
from model import *
from nltk_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as file:
    intents = json.load(file)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']
IGNORE_WORDS = data['ignore_words']

model = FeedForwardNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Fren'
user_name = input("What's your name : ")
print("Let's chat, {}. Type 'quit' to exit".format(user_name))
while True:
    sentence = input("{} : ".format(user_name))
    if sentence.strip().lower() == 'quit':
        break

    stemmed_sentence = [stem(word) for word in tokenize(
        sentence) if word not in IGNORE_WORDS]
    x = bag_of_words(stemmed_sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)

    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]

    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print("{} : {}".format(
                    bot_name, random.choice(intent['responses'])))
    else:
        print("{} : Sorry, I do not understand.".format(bot_name))

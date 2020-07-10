import json
import numpy as np
from nltk_utils import *
from model import *
from torch.utils.data import (Dataset, DataLoader)

with open('intents.json', 'r') as file:
    intents = json.load(file)

IGNORE_WORDS = ['?', '.', '!', ',', ':', ';']

all_words = []
tags = []
xy = []
x_train = []
y_train = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        words = [stem(word) for word in words if word not in IGNORE_WORDS]
        all_words.extend(words)
        xy.append((words, tag))
all_words = sorted(set(all_words))
tags = sorted(set(tags))

for (stemmed_sentence, tag) in xy:
    bog = bag_of_words(stemmed_sentence, all_words)
    x_train.append(bog)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.num_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, ind):
        return self.x_data[ind], self.y_data[ind]

    def __len__(self):
        return self.num_samples


# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 800
loss_func = nn.CrossEntropyLoss()
optimizer_func = torch.optim.Adam

dataset = ChatDataset(x_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)

input_size = len(x_train[0])
hidden_size = 16
num_classes = len(tags)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FeedForwardNet(input_size, hidden_size, num_classes).to(device)


def train(num_epochs, learning_rate, train_loader, loss_func, optimizer_func=torch.optim.SGD):
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print("Epoch [{}] : Loss : {}".format(epoch, loss.item()))


train(num_epochs, learning_rate, train_loader, loss_func, optimizer_func)

data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': num_classes,
    'all_words': all_words,
    'tags': tags,
    'ignore_words': IGNORE_WORDS
}

FILE = 'data.pth'
torch.save(data, FILE)

print("Training complete. Model saved.")

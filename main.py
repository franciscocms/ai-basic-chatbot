import nltk
from nltk.stem.lancaster import LancasterStemmer
import torch
import torch.nn as nn
import json
import numpy as np
import random
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training_data, output_data = pickle.load(f)
    print('Data loaded...\t')
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in  data["intents"]:
        for pattern in intent['patterns']: # getting each pattern from every tag
            wrds = nltk.word_tokenize(pattern) # list of all different words in pattern
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])


    # Stemming all words in our words list

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    # -> set() removes any duplicates
    # -> after converting to a list data type again, words are sorted

    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    for x, doc in enumerate(docs_x): # each doc is one pattern
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = labels.index(docs_y[x])

        training.append(bag)
        output.append(output_row)

    training_data = np.array(training)
    output_data = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training_data, output_data), f)


class model(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(in_features = in_features, out_features = 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, n_classes)

        self.activation = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class dataset():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):  

        x = self.data[index]
        y = self.labels[index]

        # pylint: disable=E1101
        return torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))
        # pylint: enable=E1101
        
    def __len__(self):
        return len(self.data)


def train_model(model, criterion, optimizer, num_epochs):
     
    for epoch in range(num_epochs):    
        
        model.train()
        for inputs, labels in trainloader:                        
            with torch.set_grad_enabled(True):
                
                outputs = model(inputs.float())              
                loss = criterion(outputs, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            t_loss = 0.
            for i, (inputs, labels) in enumerate(trainloader):  
                outputs = model(inputs.float())
                t_loss += criterion(outputs, labels.long())
                
            t_loss /= (i + 1)

        if epoch % 10 == 0:     
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            print('....train loss: {:.3f}'.format(t_loss.item()))
            print()

    return model     

n_classes = 6
net = model(len(training_data[0]), n_classes)

trainloader = torch.utils.data.DataLoader(dataset(training_data, output_data), batch_size = 8)

try:
    net = model(len(training_data[0]), n_classes)
    net.load_state_dict(torch.load('model.pth'))
    print('Model loaded...\n')
except:
    net = train_model(net, nn.CrossEntropyLoss(), 
                        torch.optim.Adam(net.parameters(), lr = 0.01),
                        num_epochs=300)
    torch.save(net.state_dict(), 'model.pth')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

def predict(model, inputs):
    with torch.no_grad():
        model.eval()
        inputs = torch.from_numpy(inputs)
        outputs = model(inputs.float())

    return outputs.detach().numpy()

def chat():
    print("Start talking with the bot! ('quit' to stop)")
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        result = predict(net, bag_of_words(inp, words))
        results_index = np.argmax(result)
        tag = labels[results_index]

        with open("intents.json") as file:
            js_data = json.load(file)

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))

chat()
import os
import sys
from requests import get
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
from torch import optim
from sklearn.metrics import confusion_matrix

device = "cuda:2"

class CoverNotFound(Exception):
    pass

def get_cover(isbn, size="L", filename=None):
    response = get("https://covers.openlibrary.org/b/isbn/{}-{}.jpg?default=false".format(isbn, size))
    if response.status_code != 200:
        raise CoverNotFound

    if not filename:
        raise ValueError

    with open(filename, "wb") as outputfile:
        outputfile.write(response.content)

    return open(filename, "rb")

# The above is kind of useless since the kaggle challenge already contained the covers but... :D

def get_covers_table(tablefilename):
    return pd.read_csv(tablefilename, index_col=None)

class CoversDataset(Dataset):
    def __init__(self, tablefilename, imgroot, size, actualsize=1.0):
        self.covertable = get_covers_table(tablefilename).sample(frac=actualsize)
        self.imgroot = imgroot
        self.resize = Resize(size)
        
        images = []
        bad_indices = []
        for index, row in tqdm(self.covertable.iterrows(), total=len(self.covertable)):
            imagepath = os.path.join(self.imgroot, row['img_paths'])
            the_image = self.resize(read_image(imagepath)).float()
            if the_image.size()[0] != 3:
                print("rejecting image {} with bad dimensions {}".format(imagepath, the_image.size()))
                bad_indices.append(index)
                continue
            images.append(the_image)

        self.covertable.drop(bad_indices, inplace=True)

        self.allimgs = torch.stack(images).to(device)
        
        categorylist = list(set(list(self.covertable['category'])))
        categoryvals = list(range(len(categorylist)))
        self.cat2label = dict(zip(categorylist, categoryvals))

        self.covertable['label'] = [self.cat2label[x] for x in self.covertable['category']]

    def __len__(self):
        return len(self.covertable)

    def __getitem__(self, idx): 
        row = self.covertable.iloc[idx]
        imagepath = os.path.join(self.imgroot, row['img_paths'])
        label = row['label']
        image = self.allimgs[idx]
        return image, label

# Now we build our model.
class CoversGenreModel(nn.Module):
    def __init__(self, outputsize, xdim, ydim):
        super().__init__()
        self.outputsize = outputsize

        self.conv2d = torch.nn.Conv2d(3, 1, (3,3), padding=1)
        self.linear = torch.nn.Linear(xdim*ydim, outputsize)
    
    def forward(self, batch):
        output = self.conv2d(batch).squeeze(1)
        batchsize = output.size()[0]
        dim1 = output.size()[1]
        dim2 = output.size()[2]
        output = output.view((batchsize, dim1*dim2))
        output = self.linear(output)
        return torch.log_softmax(output, 1)

class CoversGenreTanhModel(CoversGenreModel):
    def __init__(self, outputsize, xdim, ydim, hidden=1000, dropout=0.01):
        super().__init__(outputsize, xdim, ydim)
        self.tanh = torch.nn.Tanh()
        self.linear0 = torch.nn.Linear(xdim*ydim, hidden)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(hidden, outputsize)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, batch):
        output = self.conv2d(batch).squeeze(1)
        batchsize = output.size()[0]
        dim1 = output.size()[1]
        dim2 = output.size()[2]
        output = output.view((batchsize, dim1*dim2))
        output = self.relu(output)
        output = self.linear0(output)
        output = self.dropout(output)
        output = self.tanh(output)
        output = self.linear(output)
        return torch.log_softmax(output, 1)

def create_env(tablefilename, imgroot, imgsize, testsize=0.3, actualsize=1.0):
    dataset = CoversDataset(tablefilename, imgroot, imgsize, actualsize)
    train_set, test_set = random_split(dataset, (1.0-testsize, testsize))    

    return train_set, test_set

def train(train_set, model, epochs=25, batch_size=50, lr=0.001):
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            X, y = batch
            y = y.to(device)
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            total_loss += float(loss)
            optimizer.step()

        print("At epoch {}, we get loss {}.".format(epoch, total_loss))

def test(test_set, model):
    with torch.no_grad():
        correct = 0
        predicted = []
        actual = []
        for item in range(len(test_set)):
            output = model(test_set[item][0])
            expected = test_set[item][1]
            output = torch.argmax(output)
            predicted.append(int(output))
            actual.append(int(expected))
            if output == expected:
                correct += 1.0
            
        
        accuracy = correct/len(test_set)
        matrix = confusion_matrix(actual, predicted)
        print("Accuracy = {}".format(accuracy))
        print("Confusion = {}".format(matrix))
        return accuracy, matrix
        

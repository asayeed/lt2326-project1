import os
import sys
from requests import get
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

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
    def __init__(self, tablefilename, imgroot, size):
        self.covertable = get_covers_table(tablefilename)
        self.imgroot = imgroot
        self.resize = Resize(size)

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
        image = self.resize(read_image(imagepath))
        return image, label

# Now we build our model.
class CoversGenreModel(nn.Module):
    def __init__(self, outputsize):
        super().__init__()
        self.outputsize = outputsize

        self.conv2d = torch.nn.Conv2d(3, 1, (3,3))
    
    def forward(self, batch):
        output = self.conv2d(batch).squeeze(1)
        batchsize = output.size[0]
        dim1 = output.size()[1]
        dim2 = output.size()[2]
        output = output.view((batchsize, dim1*dim2))
        return torch.log_softmax(output, 1)



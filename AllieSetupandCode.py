#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import cv2
import os
import torch
import pickle as pkl
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch.optim as optim

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pickle as pkl
import torch
from torchvision.utils import save_image
import cython
import os
import numpy as np

trainx = torch.load('/kaggle/input/csci-ua-473-intro-to-machine-learning-fall22/train/train/trainX.pt')
trainy = torch.load('/kaggle/input/csci-ua-473-intro-to-machine-learning-fall22/train/train/trainY.pt')
testx = torch.load('/kaggle/input/csci-ua-473-intro-to-machine-learning-fall22/test/test/testX.pt')

num_train = trainx[0].shape[0]
num_test = testx[0].shape[0]

os.makedirs('./lazydata', exist_ok=True)

# Save train data
os.makedirs('./lazydata/train', exist_ok=True)
os.makedirs('./lazydata/train/X', exist_ok=True)
os.makedirs('./lazydata/train/Y', exist_ok=True)
for i in range(num_train):
    os.makedirs('./lazydata/train/X/{}'.format(i), exist_ok=True)
    # rgb
    os.makedirs('./lazydata/train/X/{}/rgb'.format(i), exist_ok=True)
    for j in range(3):
        save_image(trainx[0][i][j]/255, './lazydata/train/X/{}/rgb/{}.png'.format(i, j))
    # depth
    depth = trainx[1][i].numpy()
    np.save('./lazydata/train/X/{}/depth.npy'.format(i), depth)
    # field id
    pkl.dump(trainx[2][i], open('./lazydata/train/X/{}/field_id.pkl'.format(i), 'wb'))

    y = trainy[0][i].numpy()
    np.save('./lazydata/train/Y/{}.npy'.format(i), y)
print("Saved train data")

# Save test data
os.makedirs('./lazydata/test', exist_ok=True)
os.makedirs('./lazydata/test/X', exist_ok=True)
for i in range(num_test):
    os.makedirs('./lazydata/test/X/{}'.format(i), exist_ok=True)
    # rgb
    os.makedirs('./lazydata/test/X/{}/rgb'.format(i), exist_ok=True)
    for j in range(3):
        save_image(testx[0][i][j]/255, './lazydata/test/X/{}/rgb/{}.png'.format(i, j))
    # depth
    depth = testx[1][i].numpy()
    np.save('./lazydata/test/X/{}/depth.npy'.format(i), depth)
    # field id
    pkl.dump(testx[2][i], open('./lazydata/test/X/{}/field_id.pkl'.format(i), 'wb'))

print("Saved test data")

class LazyLoadDataset(Dataset):
    def __init__(self, path, train = True, transform = None):
        self.transform = transform
        self.train = train
        
        path = path + ('train/' if train else 'test/')

        self.pathX = path + 'X/'
        self.pathY = path + 'Y/'

        self.data = os.listdir(self.pathX)
    
    def __getitem__(self, index):
        f = self.data[index]

        x = cv2.imread(self.pathX + f + '/rgb/0.png')
        y = cv2.imread(self.pathX + f + '/rgb/1.png')
        z = cv2.imread(self.pathX + f + '/rgb/2.png')

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
            z = self.transform(z)
        
        depth = np.load(self.pathX + f + '/depth.npy') /1000

        field_id = pkl.load(open(self.pathX + f + '/field_id.pkl', 'rb'))
        
        if self.train:
            Y = np.load(self.pathY + f + '.npy')

            return (x, y, z, depth, field_id), Y
        else:
            return (x, y, z, depth, field_id)
        
    def __len__(self):
        return len(self.data)
    
dataset = LazyLoadDataset('./lazydata/', transform = transforms.Compose([transforms.ToTensor(), ]))
loader_for_norm = DataLoader(dataset, batch_size=len(dataset), shuffle=True)


# In[ ]:


def normalize_calc(loader):
    
    (x, y, z, depth, field_id), Y = next(iter(loader))
    
    x = [0,2,3]
    
    mean0 = x.mean(x)
    mean1 = y.mean(x)
    mean2 = z.mean(x)
    
    std0 = x.std(x)
    std1 = y.std(x)
    std2 = z.std(x)
    
    sum_mean = mean0 + mean1 + mean2
    sum_std = std0 + std1 + std2
    
    mean = sum_mean/3
    std = sum_std/3
    
    return mean, std 

mean, std = normalize_calc(traindl)

normalized_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std),])
dataset = LazyLoadDataset('./lazydata/', transform = normalized_transform)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.75 * len(dataset)), int(0.25 * len(dataset))])

traindl = DataLoader(train_dataset, batch_size=32, shuffle=True)
testdl = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


for (x, y, z, depth, field_id), Y in traindl:
    for i in range(10):
        plt.imshow(z[i].squeeze().permute(1, 2, 0))
        plt.subplot(2, 5, i+1)
    plt.show()
    break

plt.show()

for (x, y, z, depth, field_id), Y in testdl:
    for i in range(10):
        plt.imshow(z[i].squeeze().permute(1, 2, 0))
        plt.subplot(2, 5, i+1)
    plt.show()
    break

plt.show()


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MYCNN(nn.Module):
    def __init__(self, input_channels, conv_features, fc_features, output_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, conv_feature, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_feature, conv_feature, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(conv_feature * 8 * 8, fc_feature)
        self.fc2 = nn.Linear(fc_feature, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size= 2, stride= 2)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)
        x = x.view(-1, self.fc1.in_features)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    


# In[ ]:


def train(epoch, model, optimizer):

    model.train()

    for batch_idx, ((x, y, z, depth, field_id), target) in enumerate(traindl):
        data = torch.cat((x, y, z, depth), dim=1).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        optimizer.step()

    return print("Training Done~")

def test(model):
    model.eval()
    test_loss = 0

    for batch_idx, ((x, y, z, depth, field_id), target) in enumerate(testdl):
        data = torch.cat((x, y, z, depth), dim=1).to(device)
        target = target.to(device)
        pred = model(data)
        
    return print("Testing Done~")

def predict(model):
    outfile = 'submission3.csv'
    output_file = open(outfile, 'w')
    titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
    
    model.eval()
    pred = []
    file_ids = []

    for i, ((x, y, z, depth, field_id)) in enumerate(testdl):
        data = torch.cat((x, y, z, depth), dim=1).to(device)
        output = model(data)
        pred.append(output.cpu().detach().numpy())
        file_ids.extend(field_id)
    
    pred = np.concatenate(pred) / 1000

    df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(pred)], axis = 1, names = titles)
    df.columns = titles
    df.to_csv(outfile, index = False)
    print("Written to csv file {}".format(outfile))

model = MYCNN(input_channels = 12, conv_features = 256, fc_features = 4096, output_size= 12)
model.to(device)

for epoch in range(0, 5):
    train(epoch, model, optim.Adam(model_cnn.parameters(), lr = 0.01))
    test(model) 


# In[ ]:


testds = LazyLoadDataset('./lazydata/', train = False, transform = transform_with_normalization)
testdl = DataLoader(testds, batch_size=32 * 2, shuffle=False)
predict(model_cnn)


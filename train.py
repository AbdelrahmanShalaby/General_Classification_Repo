import numpy
import argparse

import numpy as np
import torch
import torch.nn as nn
from data.load_custom_dataset import CustomImageDataset
from sklearn.utils import shuffle
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
from tqdm import tqdm
from torchsummary import summary
from PIL import Image
from test import test

def train(opt):


    # read csv file
    df = pd.read_csv(opt.train_csv)

    # shuffle dataframe
    df = shuffle(df)

    # take percentage 20% for validation from each class
    val = df.groupby('label').sample(frac=0.2)

    # take remaining data 80% for training
    train = df[~(df.isin(val)['image'].values)]

    # create transforms
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    # create custom dataset
    train = CustomImageDataset(df=train, img_dir=opt.imgs_path, transform=transform)
    val = CustomImageDataset(df=val, img_dir=opt.imgs_path, transform=transform)


    # generate batches from data for training and validation
    train = DataLoader(dataset= train, batch_size= opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)
    val = DataLoader(dataset= val, batch_size= opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)

    # create model architecture
    model = CNN(opt.img_channels, opt.nc, 32).to(opt.device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)

    # display model architecture
    print(model)

    # display model summary
    print(summary(model, (3, 224, 224)))

    best_loss = 1

    for epoch in range(opt.epochs):

        print("Epoch[{}/{}]:".format(epoch, opt.epochs-1))
        model.train()

        for x, y in tqdm(train, total=len(train)):
            x = x.to(device=opt.device)
            y = y.to(device=opt.device)
            # y = (torch.nn.functional.one_hot(y, opt.nc).float()).to(device=opt.device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            prediction = model(x)

            # Calculate loss
            loss = loss_fn(prediction, y)

            # Calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        best_loss = test(opt, val, model, loss_fn, best_loss)













if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Classification',
                                     description="Classification between two or more classes")
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size')
    parser.add_argument('--nc', type=int, default=2, help='number of classes')
    parser.add_argument('--img-channels', type=int, default=3, help='image channel (ex: 3 for color and 1 for gray scale)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default= 0.0001, help='value of learning rate during training')
    parser.add_argument('--img-size', type=int, default=224, help='[train, test] image sizes')
    parser.add_argument('--train-csv', type=str, default='', help='csv file contains image name and label for train')
    parser.add_argument('--test-csv', type=str, default='', help='csv file contains image name and label for test')
    parser.add_argument('--imgs-path', type=str, default='', help='folder path containing all training and testing images')
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. cuda:0 or 0,1,2,3 or cpu')
    parser.add_argument('--shuffle', type=str, default='True', help='shuffle data')
    parser.add_argument('--save-model-path', type=str, default='./models', help='path to save model')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')


    opt = parser.parse_args()

    train(opt)


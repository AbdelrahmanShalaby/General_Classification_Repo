from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
import os
from save_load_model import load_model
from model import CNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.load_custom_dataset import CustomImageDataset
import pandas as pd
import argparse


def inference(opt):

    # create model architecture
    model = CNN(opt.img_channels, opt.nc, 32).to(opt.device)

    #load model
    model = load_model(model,  os.path.join(opt.save_model_path, 'best_weights.pt'))

    # load test dataset
    test = pd.read_csv(opt.test_csv)
    test_pred = pd.DataFrame()
    test_pred['id'] = test['image'].values

    # create transforms
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # create custom dataset
    test = CustomImageDataset(df=test, img_dir=opt.imgs_path, transform=transform, train=False)
    test = DataLoader(dataset=test, batch_size=opt.batch_size, num_workers=opt.workers)

    predictions = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(test, total=len(test)):
            x = x.to(device=opt.device)

            prediction = model(x)
            predictions += list(torch.argmax(prediction, dim=1).to('cpu').numpy())

    test_pred['label'] = predictions
    test_pred.to_csv('./submission.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Classification',
                                     description="Classification between two or more classes")
    parser.add_argument('--nc', type=int, default=2, help='number of classes')
    parser.add_argument('--img-channels', type=int, default=3, help='image channel (ex: 3 for color and 1 for gray scale)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. cuda:0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-model-path', type=str, default='./models', help='path to save model')
    parser.add_argument('--test-csv', type=str, default='', help='csv file contains image name and label for test')
    parser.add_argument('--img-size', type=int, default=224, help='[train, test] image sizes')
    parser.add_argument('--imgs-path', type=str, default='', help='folder path containing all training and testing images')
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size')
    opt = parser.parse_args()

    inference(opt)



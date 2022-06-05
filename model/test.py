import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import models
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import os
import random
from efficientnet_pytorch import EfficientNet
import csv

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

data_path = "/home/users/person/hsuck/ML-Competition/dataset/training"

class MyDataset(Dataset):
    def __init__(self, x, y, z):
        self.data = x
        self.label = y
        self.filename = z

    def __getitem__(self,index):
        return self.data[index], self.label[index], self.filename[index]

    def __len__(self):
        return len(self.data)

train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# read data
import cv2
import glob
testx = []
testy = []
testz = []
for folder in glob.iglob(f'{data_path}/val/*'):
    #print(folder)
    #input('>')
    for image in os.listdir( folder ):
        img_pil = Image.open(os.path.join(folder, image), mode='r')
        tmp = train_transform(img_pil)
        testx.append( tmp )
        testy.append(torch.tensor(int(folder.split('/')[-1])))
        testz.append( image )

# make data loader
print('Test length: ' + str(len(testx)))
batch_size = 64
my_dataset_test = MyDataset(testx, testy, testz)
test_dataloader = DataLoader(my_dataset_test, shuffle=True)

model_path = '/home/users/person/hsuck/ML-Competition/model/runs/Jun04_16-57-12_gslave01/best-model.pt'
model_para_path = '/home/users/person/hsuck/ML-Competition/model/runs/Jun04_16-57-12_gslave01/best-model-parameters.pt'
model = torch.load( model_path )
state_dict = torch.load( model_para_path )
model.load_state_dict(state_dict)
print(f'resume model from {model_path}')

csvfile = open( './submission.csv', 'w', newline = '' )
writer = csv.writer( csvfile )
writer.writerow(['filename', 'category'])

model.eval()

correct = 0
total = 0
all_preds = torch.tensor([])
all_labels = torch.tensor([])

for image, label, filename in test_dataloader:
    all_labels = torch.cat( ( all_labels, label ), dim=0 )
    image = image.cuda()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    all_preds = torch.cat( ( all_preds, predicted.cpu() ), dim=0 )

    total += label.size(0)
    correct += (predicted == label.cuda()).sum()
    writer.writerow([str( filename[0] ), int( predicted.cpu() )])
    print( str( filename[0] ), int( predicted.cpu() ) )
"""
stacked = torch.stack( ( all_preds, all_labels ), dim=1 )
#print( all_preds )
#print( all_labels )
cf_matrix = np.zeros( shape = ( 219, 219 ) )
for p in stacked:
    pl, tl = p.tolist()
    #print( int(tl), int(pl) )
    cf_matrix[int(tl), int(pl)] += 1

#precision = np.zeros( shape = ( 1, 219 ) )
#recall = np.zeros( shape = ( 1, 219 ) )
Macro_F1_score = 0
for i in range( 219 ):
    TP = int( cf_matrix[i, i] )
    FP = 0
    FN = 0
    for j in range( 219 ):
        if i == j:  continue
        FP += int( cf_matrix[j, i] )
        FN += int( cf_matrix[i, j] )

    precision = TP / ( TP + FP ) if TP + FP != 0 else 0
    recall = TP / ( TP + FN ) if TP + FN != 0 else 0
    Macro_F1_score += 2 * precision * recall / ( precision + recall ) if precision + recall != 0 else 0

Macro_F1_score /= 219

acc = (100 * float(correct) / total)

final_score = 0.5 * acc + 0.5 * Macro_F1_score

print('Accuracy of test images: %f %%' % acc)
print('Final score of test images: %f' % final_score)
"""

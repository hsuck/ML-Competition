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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

data_path = "/home/users/person/hsuck/ML-Competition/data"

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self,index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform2 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip (p=1),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform3 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomVerticalFlip (p=1),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform4 = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.CenterCrop(299),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform5 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomRotation(60, interpolation=InterpolationMode.BICUBIC, expand=False),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform6 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.GaussianBlur(7,3),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform7 = transforms.Compose([
    transforms.Resize((199, 199)),
    transforms.Pad((50), fill=0, padding_mode="constant"),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# read label
import csv
labels = []
with open(data_path + '/label.csv', newline='') as csvfile:
  rows = csv.DictReader(csvfile)
  for row in rows:
    labels.append(row)

# read data
import cv2
import glob
datax = []
datay = []
test_n = 400
testx = []
testy = []
t = 0
for images in glob.iglob(f'{data_path}/*'):
    # check if the image ends with png
    if (images.endswith(".jpg")):
        for label in labels:
            if label['filename'] in images:
                img_pil = Image.open(images, mode='r')
                tmp = train_transform(img_pil)
                tmp2 = train_transform2(img_pil)
                tmp3 = train_transform3(img_pil)
                tmp4 = train_transform4(img_pil)
                tmp5 = train_transform5(img_pil)
                tmp6 = train_transform6(img_pil)
                tmp7 = train_transform7(img_pil)
                if t % 4 == 0 and t < test_n*10:
                    testx.append(tmp)
                    testy.append(torch.tensor(int(label['category'])))
                else:
                    datax.append(tmp)
                    datay.append(torch.tensor(int(label['category'])))
                t += 1
                datax.append(tmp2)
                datax.append(tmp3)
                datax.append(tmp4)
                datax.append(tmp5)
                datax.append(tmp6)
                datax.append(tmp7)
                datay.append(torch.tensor(int(label['category'])))
                datay.append(torch.tensor(int(label['category'])))
                datay.append(torch.tensor(int(label['category'])))
                datay.append(torch.tensor(int(label['category'])))
                datay.append(torch.tensor(int(label['category'])))
                datay.append(torch.tensor(int(label['category'])))

# make data loader
print('Train length: ' + str(len(datax)))
print('Test length: ' + str(len(testx)))
batch_size = 64
my_dataset_train = MyDataset(datax, datay)
my_dataset_test = MyDataset(testx, testy)
test_dataloader = DataLoader(my_dataset_test, shuffle=True)

# inception v3
model = models.inception_v3(pretrained=True)

model.aux_logits = False

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 2048),
    nn.Linear(2048, 2048),
    nn.Linear(2048, 219)
)


model = model.cuda()
loss = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
optimizer = torch.optim.SGD((filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, momentum=0.7)
#optimizer = torch.optim.RMSprop((filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, alpha=0.9)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

# SummaryWriter
writer = SummaryWriter()

num_epochs = 32
best_accuracy = 0

model.train()

print('Start training')
for epoch in range(num_epochs):
    random.seed(epoch)
    
    total_batch = len(datax)//batch_size
    
    train_dataloader = DataLoader(my_dataset_train, batch_size, shuffle=True)
    # train
    for i, (batch_images, batch_labels) in enumerate(train_dataloader):

        X = batch_images.cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()

        optimizer.step()
        #lr_scheduler.step()

        train_loss = cost.item()
        writer.add_scalar('Loss/train', train_loss, epoch+1)

        if (i+1) % batch_size == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

    # test
    correct = 0
    total = 0
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    
    model.eval()
    for images, labels in test_dataloader:
        all_labels = torch.cat( ( all_labels, labels ), dim=0 )
        images = images.cuda()

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

        all_preds = torch.cat( ( all_preds, predicted.cpu() ), dim=0 )

        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    
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

    writer.add_scalar('Acc/val', acc, epoch+1)
    print('Accuracy of test images: %f %%' % acc)
    print('Final score of test images: %f' % final_score)

    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(model, 'best-model.pt')
        torch.save(model.state_dict(), 'best-model-parameters.pt')



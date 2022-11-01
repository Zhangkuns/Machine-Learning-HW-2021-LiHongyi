# Preparing Data
# Load the training and testing data from the `.npy` file (NumPy array).

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc  # Garbage Collector interface
import torch
import torch.nn as nn

print('Loading data ...')

data_root = './timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))

# Digit Preprocess
'''def addNoise(data, p=0.02):
    return np.random.normal(0, 1, len(data)) * p + data


def centralPhone(df, noise=False):
    return np.array([i if noise else addNoise(i) for i in df[:, 39 * 4:39 * 9]])


train = centralPhone(train)
test = centralPhone(test)'''


# Create Dataset
class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


'''Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change 
the ratio of validation data. '''

VAL_RATIO = 0.05

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

'''Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here.'''

BATCH_SIZE = 128
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

'''Cleanup the unneeded variables to save memory.
notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later
the data size is quite huge, so be aware of memory usage in colab'''

'''This module provides an interface to the optional garbage collector. It provides the ability to disable the 
collector, tune the collection frequency, and set debugging options. It also provides access to unreachable objects 
that the collector found but cannot free. '''

del train, train_label, train_x, train_y, val_x, val_y

'''gc.collect(generation=2) With no arguments, run a full collection. The optional argument generation may be an 
integer specifying which generation to collect (from 0 to 2). A ValueError is raised if the generation number is 
invalid. The number of unreachable objects found is returned '''

gc.collect()

# Create Model
# Define model architecture, you are encouraged to change and experiment with the model architecture.
# DNN module
'''class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bn0 = nn.BatchNorm1d(195)
        self.layer1 = nn.Linear(195, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.layer2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.layer3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.layer4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.layer6 = nn.Linear(64, 39)

        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.bn0(x)

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer5(x)
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer6(x)

        return x'''


# RNN GRU module 因为发音事件本身取决于前后文，一个词语会有多个相同音节，采用前后文输入相关的RNN模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lstm = nn.GRU(39, 256, 2, batch_first=True, dropout=0.2)

        self.layers = nn.Sequential(
            nn.Linear(11 * 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 39)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(-1, 11, 39)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.layers(x)

        return x


# Training

# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Fix random seeds for reproducibility.
# fix random seed

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Feel free to change the training parameters here.
# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 50  # number of training epoch
learning_rate = 0.0001  # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    best_loss = 1

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest
                # probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss / len(val_loader)
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))

            # if the model doesn't improve, stop the training
            if val_loss / len(val_loader) - best_loss > 0.5:
                break
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

# Testing
# Create a testing dataset, and load model from the saved checkpoint.

# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

# Make prediction.
predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

# Write prediction to a CSV file.
# After finish running this block, download the file `prediction.csv` from the files section on the
# left-hand side and submit it to Kaggle.
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))

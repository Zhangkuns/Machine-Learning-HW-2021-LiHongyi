# Import Some Packages

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pprint as pp

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'  # path to testing data

# For seed
'''Completely reproducible results are not guaranteed across PyTorch releases, individual commits, 
   or different platforms.However, there are some steps you can take to limit the number of sources 
   of nondeterministic behavior for a specific platform, device, and PyTorch release. '''

''' First, you can control sources of randomness that can cause multiple executions of 
    your application to behave differently. '''
myseed = 42069  # set a random seed for reproducibility
np.random.seed(myseed)  # If you are using rely on NumPy,  you can seed the global NumPy RNG with:
torch.manual_seed(myseed)  # torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

'''While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects the same 
algorithm each time an application is run, that algorithm itself may be nondeterministic, unless either 
torch.use_deterministic_algorithms(True) or torch.backends.cudnn.deterministic = True is set. '''
torch.backends.cudnn.deterministic = True

'''Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False 
causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.'''
torch.backends.cudnn.benchmark = False


# **Some Utilities**


def get_device():
    """ Get device (if GPU is available, use GPU) """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    """ Plot learning curve of your DNN (train & dev loss) """
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    """ Plot prediction of your DNN """
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


# Preprocess
'''We have three kinds of datasets:
   `train`: for training
   `dev`: for validation
   `test`: for testing (w/o target value)'''

# **Dataset**
'''The `COVID19Dataset` below does:
   read `.csv` files
   extract features
   split `covid.train.csv` into train/dev sets
   normalize features'''

'''Creating a Custom Dataset
   A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. '''


class COVID19Dataset(Dataset):
    """ Dataset for loading and preprocessing the COVID19 dataset """

    '''The __init__ function is run once when instantiating the Dataset object. 
    We initialize the directory containing the images, the annotations file, and both transforms '''

    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # Read data into numpy arrays

        '''with语句的基本结构
           with expression [as variable]:
                 with-block
        expression要返回一个支持上下文管理协议的对象，后面分句as存在时，此对象返回值赋予给variable'''
        with open(path, 'r') as fp:  # ‘r'只读模式
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)  # 去除第一行和第一列

        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            feats = list(range(40)) + [40, 41, 42, 43, 57, 58, 59, 60, 61, 75, 76, 77, 78, 79]
            pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev(validation) sets
            # 更推荐使用打乱后的处理的随机分组
            indices_tr, indices_dev = train_test_split([i for i in range(data.shape[0])], test_size=0.3, random_state=0)
            if self.mode == 'train':
                indices = indices_tr
            elif self.mode == 'dev':
                indices = indices_dev

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        '''特征标准化：
           此处是对数据进行标准化处理，可以将不同量纲的不同特征，变为同一个数量级，使得损失函数更加平滑
           标准化的优点：①提升模型的精度   ②提升收敛速度
           采用均值标准化： （第i维数据 - 第i维数据的平均值）/（第i维数据的标准差）'''
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)  # std:Compute the standard deviation along the specified axis.

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


# DataLoader
# A `DataLoader` loads data from a given `Dataset` into batches.
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    """ Generates a dataset, then is put into a dataloader. """
    # Construct dataset
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    # Construct dataloader
    '''DataLoader(dataset, batch_size, shuffle=False, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
        
        shuffle (bool, optional):
        set to True to have the data reshuffled at every epoch (default: False)
        
        drop_last (bool, optional):(default: False)
        set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
         
        num_workers:   
        Setting the argument num_workers as a positive integer will turn on multi-process data loading with the 
        specified number of loader worker processes.
        
        pin_memory (bool, optional):
        If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. 
        If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
        '''
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=n_jobs,
                            pin_memory=True)
    return dataloader


# **Deep Neural Network**
# `NeuralNet` is an `nn.Module` designed for regression.
# The DNN consists of 2 fully-connected layers with ReLU activation.
# This module also included a function `cal_loss` for calculating loss.

class NeuralNet(nn.Module):
    """ A simple fully-connected deep neural network """
    '''We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__. 
       Every nn.Module subclass implements the operations on input data in the forward method.'''

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')  # 均方损失函数

    def forward(self, x):
        """ Given input of size (batch_size x input_dim), compute output of the network """
        return self.net(x).squeeze(1)  # 对数据的维度进行压缩，方便预测值与实际值的对比

    def cal_loss(self, pred, target):
        """ Calculate loss """
        # TODO: you may implement L2 regularization here
        return torch.sqrt(self.criterion(pred, target))


# Training

def train(tr_set, dv_set, model, config, device):
    """ DNN training """

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    '''To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variables) 
       to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.'''
    '''getattr() 函数用于返回一个对象属性值。
       getattr(object, name[, default])'''
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()  # set model to training mode
        for x, y in tr_set:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            '''loss.backward()故名思义，就是将损失loss 向输入侧进行反向传播，同时对于需要进行梯度计算的所有变量 x (requires_grad=True)，
            计算梯度 dloss/dx ，并将其累积到梯度x.grad中备用，即：x.grad = x.grad + dloss/dx'''
            mse_loss.backward()  # compute gradient (backpropagation)
            '''optimizer.step()是优化器对 x 的值进行更新，以随机梯度下降SGD为例：学习率(learning rate, lr)来控制步幅，
            即：x = x-lr*x.grad ，减号是由于要沿着梯度的反方向调整变量值以减少Cost。'''
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


# Validation

def dev(dv_set, model, device):
    model.eval()  # set model to evaluation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # compute averaged loss

    return total_loss


# Testing

def test(tt_set, model, device):
    model.eval()  # set model to evaluation mode
    preds = []
    for x in tt_set:  # iterate through the dataloader 计算获得【预测值】
        x = x.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction detach会分离出一个新的tensor，这个tensor不能够求导
    '''Concatenates the given sequence of seq tensors in the given dimension. 
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.'''
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds


# **Setup Hyper-parameters**
# `config` contains hyper-parameters for training and the path to save your model.
device = get_device()  # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
# target_only = False  # TODO: Using 40 states & 2 tested_positive features
target_only = True
# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,  # maximum number of epochs
    'batch_size': 270,  # mini-batch size for dataloader
    'optimizer': 'SGD',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,  # learning rate of SGD
        'momentum': 0.9  # momentum for SGD
    },
    'early_stop': 200,  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

# **Load data and model**

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

# **Start Training!**
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set


# **Testing**

def save_pred(preds, file):
    """ Save predictions to specified file """
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')  # save prediction file to pred.csv

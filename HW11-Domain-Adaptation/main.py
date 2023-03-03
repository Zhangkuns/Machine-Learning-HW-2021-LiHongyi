# Homework 11 - Transfer Learning (Domain Adversarial Training)


# Special Domain Knowledge

"""
When we graffiti, we usually draw the outline only, therefore we can perform edge detection processing
on the source data to make it more similar to the target data.
"""

# Canny Edge Detection
'''
The implementation of Canny Edge Detection is as follow.
The algorithm will not be describe thoroughly here.  If you are interested, please refer to the wiki or [here]
(https://medium.com/@pomelyu5199/canny-edge-detector-%E5%AF%A6%E4%BD%9C-opencv-f7d1a0a57d19).

We only need two parameters to implement Canny Edge Detection with CV2:  `low_threshold` and `high_threshold`.

```cv2.Canny(image, low_threshold, high_threshold)```

Simply put, when the edge value exceeds the high_threshold, we determine it as an edge. 
If the edge value is only above low_threshold, we will then determine whether it is an edge or not.

Let's implement it on the source data.'''

# Data Process
# The data is suitable for `torchvision.ImageFolder`.You can create a dataset with `torchvision.ImageFolder`.
# Details for image augmentation please refer to the comments in the following codes.

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn import manifold
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

source_transform = transforms.Compose([
    # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
    transforms.Grayscale(),
    # cv2 do not support skimage.Image, so we transform it to np.array,
    # and then adopt cv2.Canny algorithm.
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # Transform np.array back to the skimage.Image.
    transforms.ToPILImage(),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # Turn RGB to grayscale.
    transforms.Grayscale(),
    # Resize: size of source data is 32x32, thus we need to
    #  enlarge the size of target data from 28x28 to 32x32。
    transforms.Resize((32, 32)),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing2022/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing2022/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)


# Model
# Feature Extractor: Classic VGG-like architecture
# Label Predictor / Domain Classifier: Linear models.

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


# Pre-processing
# Here we use Adam as our optimizor.

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

# Start Training

# DaNN Implementation
'''
In the original paper, Gradient Reversal Layer is used.
Feature Extractor, Label Predictor, and Domain Classifier are all trained at the same time. 
In this code, we train Domain Classifier first, and then train our Feature Extractor 
(same concept as Generator and Discriminator training process in GAN).
'''
# Reminder
'''
* Lambda, which controls the domain adversarial loss, is adaptive in the original paper. You can refer to 
  [the original work](https://arxiv.org/pdf/1505.07818.pdf) . Here lambda is set to 0.1.
* We do not have the label to target data, you can only evaluate your model by uploading your result to kaggle.:)
'''


def train_epoch(source_dataloader, target_dataloader, lamb):
    """
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptation and classification.
    """

    # D loss: Domain Classifier的loss
    # F loss: Feature Extractor & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()

        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (running mean/var of source and target data are different.)

        # torch.cat(tensors, dim=0, *, out=None) → Tensor
        # Concatenates the given sequence of seq tensors in the given dimension.
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        mixed_data = torch.cat([source_data, target_data], dim=0)

        # torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
        # Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)

        # We don't need to train feature extractor in step 1.
        # Thus, we detach the feature neuron to avoid backpropagation.
        # Tensor.detach()
        # Returns a new Tensor, detached from the current graph.
        # The result will never require gradient.
        # This method also affects forward mode AD gradients and the result will never have forward mode AD gradients.
        domain_logits = domain_classifier(feature.detach())

        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num


# train 200 epochs
epochs = 3000
marked_epoch = [1, epochs // 3, epochs - 1]
for epoch in range(epochs):
    # 使用论文中的lamb参数 lamb = (2/(e^(-10epoch/epochs)+1))-1
    train_D_loss, train_F_loss, train_acc = \
        train_epoch(source_dataloader, target_dataloader, lamb=(2/(math.exp(-10*epoch/epochs)+1))-1)

    if epoch in marked_epoch:
        torch.save(feature_extractor.state_dict(), f'extractor_model_{epoch}.bin')
        torch.save(label_predictor.state_dict(), f'predictor_model_{epoch}.bin')

    torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss,
                                                                                           train_F_loss, train_acc))

# Inference

# We use pandas to generate our csv file.
# By The Way, the performance of the model trained for 200 epoches might be unstable.
# You can train for more epoches for a more stable performance.

result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
df.to_csv('DaNN_submission.csv', index=False)


# Visualization

# We use t-SNE plot to observe the distribution of extracted features.

class Feature:
    def __init__(self):
        self.X = []
        self.TX = []
        self.labels = []


# Step1: Load checkpoint and evaluate to get extracted features

# Hints:
# Set features_extractor to eval mode
# Start evaluation and collect features and labels

def get_features(model_list):
    features = []
    for model in model_list:
        model.cuda()
        features.append(Feature())

    for (x, y), (tx, _) in zip(source_dataloader, target_dataloader):
        x, tx = x.cuda(), tx.cuda()
        for i, model in enumerate(model_list):
            features[i].X.append(model(x).detach().cpu())
            features[i].TX.append(model(tx).detach().cpu())
            features[i].labels.append(y)

    for feature in features:
        feature.X = torch.cat(feature.X).numpy()
        feature.TX = torch.cat(feature.TX).numpy()
        feature.labels = torch.cat(feature.labels).numpy()

    return features


def visualization(features):

    # Step2: Apply t-SNE and normalize

    # process extracted features with t-SNE
    # X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

    # Normalization the processed features
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)

    for i, feature in enumerate(features):
        X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(feature.X)
        TX_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(feature.TX)
        # Normalization the processed features
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        tx_min, tx_max = TX_tsne.min(0), TX_tsne.max(0)
        TX_norm = (TX_tsne - tx_min) / (tx_max - tx_min)

        # Step3: Visualization with matplotlib
        # Data Visualization
        # Use matplotlib to plot the distribution
        # The shape of X_norm is (N,2)

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.title(f'stage {i}:distribution of features across different class')

        # matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None,
        # alpha=None, line widths=None, *, edge-colors=None, plotnonfinite=False, data=None, **kwargs)[source]
        # A scatter plot of y vs. x with varying marker size and/or color.
        '''
        x, y:float or array-like, shape (n, )
        s:float or array-like, shape (n, ), optional
          The marker size in points**2 (typographic points are 1/72 in.).
        c:array-like or list of colors or color, optional
          The marker colors.
        '''
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c=feature.labels)
        plt.subplot(122)
        plt.title(f'stage {i}:distribution of features across different domain')
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c='b', label='source domain')
        plt.scatter(TX_norm[:, 0], TX_norm[:, 1], c='r', label='target domain')
        # plt.legend()函数的作用是给图像加图例
        plt.legend()
    plt.show()


model_list = []
for epoch in marked_epoch:
    model = FeatureExtractor()
    model.load_state_dict(torch.load(f'extractor_model_{epoch}.bin'))
    model_list.append(model)

visualization(get_features(model_list))






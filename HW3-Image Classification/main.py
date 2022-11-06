# **Homework 3 - Convolutional Neural Network**

"""This is the example code of homework 3 of the machine learning course by Prof. Hung-yi Lee.

In this homework, you are required to build a convolutional neural network for image classification, possibly with
some advanced training tips.


There are three levels here:

**Easy**: Build a simple convolutional neural network as the baseline. (2 pts)

**Medium**: Design a better architecture or adopt different data augmentations to improve the performance. (2 pts)

**Hard**: Utilize provided unlabeled data to obtain better results. (2 pts)"""

# **Import Packages**

'''First, we need to import packages that will be used later.

In this homework, we highly rely on **torchvision**, a library of PyTorch.'''

# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import os
# This is for the progress bar.
from tqdm.auto import tqdm
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
import gc

# **About the Dataset**

_exp_name = "sample"

'''The dataset used here is food-11, a collection of food images in 11 classes.

For the requirement in the homework, TAs slightly modified the data.
Please DO NOT access the original fully-labeled training data or testing labels.

Also, the modified dataset is for this course only, and any further distribution or commercial use is forbidden.'''

# **Preprocess Data
# use transformer method to enlarge the train data

Path = 'D:/PycharmProjects/HW3-Image Classification/food-11/training/labeled'
label_set_number = 0
n_label_set_numbers = 10
in_label_number = 0
label_number = 0


def n_in_label_numbers(label_set_number):
    if label_set_number == 0:
        return 993
    if label_set_number == 1:
        return 428
    if label_set_number == 2:
        return 1499
    if label_set_number == 3:
        return 985
    if label_set_number == 4:
        return 847
    if label_set_number == 5:
        return 1324
    if label_set_number == 6:
        return 439
    if label_set_number == 7:
        return 279
    if label_set_number == 8:
        return 854
    if label_set_number == 9:
        return 1499
    if label_set_number == 10:
        return 708


def read_PIL(image_path):
    """ read image in specific path
    and return PIL.Image instance"""
    image = Image.open(image_path)
    return image


# 水平翻转
def horizontal_flip(image):
    HF = transforms.RandomHorizontalFlip()
    hf_image = HF(image)
    return hf_image


# 垂直翻转
def vertical_flip(image):
    VF = transforms.RandomVerticalFlip()
    vf_image = VF(image)
    return vf_image


# 90°角度旋转
def r90_rotation(image):
    R90 = transforms.RandomRotation([0, 30])
    r90_image = R90(image)
    return r90_image


# 270°角度旋转
def r270_rotation(image):
    R270 = transforms.RandomRotation([330, 360])
    r270_image = R270(image)
    return r270_image


# 水平翻转后再90°角度旋转
def horizontal_r90_rotation(image):
    horizontal_r90_rotation_method = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation([0, 30])
    ])
    horizontal_r90 = horizontal_r90_rotation_method(image)
    return horizontal_r90


# 水平翻转后再180°角度旋转
def horizontal_r180_rotation(image):
    horizontal_r180_rotation_method = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation([180, 180])
    ])
    horizontal_r180 = horizontal_r180_rotation_method(image)
    return horizontal_r180


# 水平翻转后再270°角度旋转
def horizontal_r270_rotation(image):
    horizontal_r270_rotation_method = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation([330, 360])
    ])
    horizontal_r270 = horizontal_r270_rotation_method(image)
    return horizontal_r270


for label_set_number in range(0, 11):
    label_number = n_in_label_numbers(label_set_number)
    if label_set_number <= 9:
        concrete_path = Path + '/0' + str(label_set_number) + '/'
        in_label_number = 0
        for in_label_number in range(0, n_in_label_numbers(label_set_number) + 1):
            if os.path.isfile(Path + '/0' + str(label_set_number) + '/' + str(label_set_number) + '_' + str(
                    in_label_number) + '.jpg'):
                image = read_PIL(Path + '/0' + str(label_set_number) + '/' + str(label_set_number) + '_' + str(
                    in_label_number) + '.jpg')

                label_number = label_number + 1
                hf_image = horizontal_flip(image)  # 水平翻转
                hf_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                vf_image = vertical_flip(image)  # 垂直翻转
                vf_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                r90_image = r90_rotation(image)  # 90°翻转
                r90_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                r270_image = r270_rotation(image)  # 270°翻转
                r270_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                horizontal_r90_image = horizontal_r90_rotation(image)  # 水平翻转后90°旋转
                horizontal_r90_image.save(
                    os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                horizontal_r180_image = horizontal_r180_rotation(image)  # 水平翻转后180°旋转
                horizontal_r180_image.save(
                    os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                horizontal_r270_image = horizontal_r270_rotation(image)  # 水平翻转后270°旋转
                horizontal_r270_image.save(
                    os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

    if label_set_number == 10:
        concrete_path = Path + '/10/'
        in_label_number = 0
        for in_label_number in range(0, n_in_label_numbers(label_set_number) + 1):
            if os.path.isfile(Path + '/10/10_' + str(in_label_number) + '.jpg'):
                image = read_PIL(Path + '/10/10_' + str(in_label_number) + '.jpg')

                label_number = label_number + 1
                hf_image = horizontal_flip(image)  # 水平翻转
                hf_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                vf_image = vertical_flip(image)  # 垂直翻转
                vf_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                r90_image = r90_rotation(image)  # 90°翻转
                r90_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                r270_image = r270_rotation(image)  # 270°翻转
                r270_image.save(os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                horizontal_r90_image = horizontal_r90_rotation(image)  # 水平翻转后90°旋转
                horizontal_r90_image.save(
                    os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                horizontal_r180_image = horizontal_r180_rotation(image)  # 水平翻转后180°旋转
                horizontal_r180_image.save(
                    os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

                label_number = label_number + 1
                horizontal_r270_image = horizontal_r270_rotation(image)  # 水平翻转后270°旋转
                horizontal_r270_image.save(
                    os.path.join(concrete_path, str(label_set_number) + '_' + str(label_number) + '.jpg'))

# **Dataset, Data Loader, and Transforms**

# **Transforms**
'''Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.

Here, since our data are stored in folders by class labels, we can directly apply 
**torchvision.datasets.DatasetFolder** for wrapping data without much effort. 

Please refer to [PyTorch official website](https://pytorch.org/vision/stable/transforms.html) for details about 
different transforms. '''

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation([-30, 30]),
    transforms.RandomVerticalFlip(p=0.1),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# **Datasets**
'''The data is labelled by the name, so we load images and label while calling '__getitem__' '''


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label


# 半监督学习的Dataset
# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
class PseudoDataset(Dataset):
    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, unlabeled_set, indices, pseudo_labels):
        self.data = Subset(unlabeled_set, indices)
        self.target = torch.LongTensor(pseudo_labels)[indices]

    def __getitem__(self, index):

        if index < 0:  # Handle negative indices
            index += len(self)
        if index >= len(self):
            raise IndexError("index %d is out of bounds for axis 0 with size %d" % (index, len(self)))

        x = self.data[index][0]
        y = self.target[index].item()
        return x, y

    def __len__(self):

        return len(self.data)


# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 32
_dataset_dir = "./food11"

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg",
                          transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

'''train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)'''

# **Model**

'''The basic model here is simply a stack of convolutional layers followed by some fully-connected layers.

Since there are three channels for a color image (RGB), the input channels of the network must be three. In each 
convolutional layer, typically the channels of inputs grow, while the height and width shrink (or remain unchanged, 
according to some hyperparameters like stride and padding). 

Before fed into fully-connected layers, the feature map must be flattened into a single one-dimensional vector (for 
each image). These features are then transformed by the fully-connected layers, and finally, we obtain the "logits" 
for each class. 

### **WARNING -- You Must Know** You are free to modify the model architecture here for further improvement. However, 
if you want to use some well-known architectures such as ResNet50, please make sure **NOT** to load the pre-trained 
weights. Using such pre-trained models is considered cheating and therefore you will be punished. Similarly, 
it is your responsibility to make sure no pre-trained weights are used if you use **torch.hub** to load any modules. 

For example, if you use ResNet-18 as your model:

model = torchvision.models.resnet18(pretrained=**False**) → This is fine.

model = torchvision.models.resnet18(pretrained=**True**)  → This is **NOT** allowed.'''


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3,128,128]  [3, 256, 256]
        self.cnn_layers = nn.Sequential(
            # 3 * 224 * 224 -> 64 * 111 * 111
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 64 * 111 * 111 -> 128 * 54 * 54
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 128 * 54 * 54 -> 256 * 26 * 26
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 256 * 26 * 26  -> 256 * 12 * 12
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 256 * 12 * 12  -> 512 * 5 * 5
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 5 * 5, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


# **Training**

'''You can finish supervised learning by simply running the provided code without any modification.

The function "get_pseudo_labels" is used for semi-supervised learning.
It is expected to get better performance if you use unlabeled data for semi-supervised learning.
However, you have to implement the function on your own and need to adjust several hyperparameters manually.

For more details about semi-supervised learning, please refer to [Prof. Lee's slides](
https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/semi%20(v3).pdf). 

Again, please notice that utilizing external data (or pre-trained model) for training is **prohibited**.'''


def get_pseudo_labels(dataset, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    print('get pseudo labels...')
    total_unlabeled = len(dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    masks = []
    pseudo_labels = []
    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        #  Filter the data and construct a new dataset.

        #  torch.max(input, dim, keepdim=False, *, out=None)
        #  Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor
        #  in the given dimension dim. And indices is the index location of each maximum value found (argmax).
        preds = torch.max(probs, 1)[1]
        mask = torch.max(probs, 1)[0] > threshold  # 过滤出满足条件的序号，即可以确定预测出来的图片

        # append用于在列表末尾添加新的对象。
        masks.append(mask)
        pseudo_labels.append(preds)

    # torch.cat(tensors, dim=0, *, out=None) → Tensor
    # Concatenates the given sequence of seq tensors in the given dimension.
    # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    masks = torch.cat(masks, dim=0).cpu().numpy()
    pseudo_labels = torch.cat(pseudo_labels, dim=0).cpu().numpy()

    # torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None,
    # requires_grad=False) → Tensor Returns a 1-D tensor of size [end-start/step] with values from the interval [
    # start, end) taken with common difference step beginning from start.
    indices = torch.arange(0, total_unlabeled)[masks]
    dataset = PseudoDataset(unlabeled_set, indices, pseudo_labels)
    print('using {0:.2f}% unlabeld data'.format(100 * len(dataset) / total_unlabeled))

    # # Turn off the eval mode.
    model.train()
    return dataset


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 200
patience = 300  # If no improvement in 'patience' epochs, early stop

# Whether to do semi-supervised learning.
do_semi = True

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, model)

        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
        print('total number of training data:', len(concat_dataset))

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        optimizer = torch.optim.Adam(model.parameters(), lr=(0.0003 / pow(epoch + 1, 1 / 2)), weight_decay=1e-5)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
# **Testing**

'''For inference, we need to make sure the model is in eval mode, and the order of the dataset should not be shuffled 
("shuffle=False" in test_loader). 

Last but not least, don't forget to save the predictions into a single CSV file.
The format of CSV file should follow the rules mentioned in the slides.

### **WARNING -- Keep in Mind**

Cheating includes but not limited to:
1.   using testing labels,
2.   submitting results to previous Kaggle competitions,
3.   sharing predictions with others,
4.   copying codes from any creatures on Earth,
5.   asking other people to do it for you.

Any violations bring you punishments from getting a discount on the final grade to failing the course.

It is your responsibility to check whether your code violates the rules.
When citing codes from the Internet, you should know what these codes exactly do.
You will **NOT** be tolerated if you break the rule and claim you don't know what these codes do.'''

# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()

# Initialize a list to store the predictions.
prediction = []

with torch.no_grad():
    for data, _ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

''''# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())'''

# Save predictions into the file.
with open("predict.csv", "w") as f:
    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in enumerate(prediction):
        f.write(f"{i},{pred}\n")

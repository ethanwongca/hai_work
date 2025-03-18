BASE_DIR = <Input>

import os
import random
import shutil
import math

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from PIL import Image

import numpy as np
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sys
sys.path.append(BASE_DIR)
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=4
MANUAL_SEED = 1
HIDDEN_SIZE = 256
INPUT_SIZE=6
utils.INPUT_SIZE = INPUT_SIZE

transform = transforms.Compose(
    [transforms.Resize((150,150)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class VTNet(nn.Module):
    def __init__(self,
                 input_size=6,
                 rnn_hidden_size=256,
                 output_size=2,
                 batch_size=4,
                 rnn_type='gru',
                 rnn_num_layers=1,
                 n_channels_1=6,
                 kernel_size_1=5,
                 n_channels_2=16,
                 kernel_size_2=5,
                 img_n_vert=150,
                 img_n_hor=150):
        """

        Args:
            input_size (int):
            hidden_size (int):
            output_size (int):
            batch_size (int):
            rnn_type (int):
            num_layers (int):
        """
        super(VTNet, self).__init__()

        self.n_channels_2 = n_channels_2


        # CNN portion
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels_1, kernel_size=kernel_size_1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n_channels_1, out_channels=n_channels_2, kernel_size=kernel_size_2, stride=1)
        # output size calculations
        self.conv1_out_vert = img_n_vert - kernel_size_1 + 1
        self.conv1_out_hor = img_n_hor - kernel_size_1 + 1
        self.mp1_out_vert = int(np.floor((self.conv1_out_vert - 2)/2) + 1)
        self.mp1_out_hor = int(np.floor((self.conv1_out_hor - 2) / 2) + 1)
        self.conv2_out_vert = self.mp1_out_vert - kernel_size_2 + 1
        self.conv2_out_hor = self.mp1_out_hor - kernel_size_2 + 1
        self.mp2_out_vert = int(np.floor((self.conv2_out_vert - 2)/2) + 1)
        self.mp2_out_hor = int(np.floor((self.conv2_out_hor - 2) / 2) + 1)
        self.fc1 = nn.Linear(n_channels_2 * self.mp2_out_hor * self.mp2_out_vert, 50)
        self.fc2 = nn.Linear(rnn_hidden_size + 50, 20)
        self.fc3 = nn.Linear(20, output_size)

        # RNN portion
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=6, num_heads=1)
        #self.multihead_attn2 = nn.MultiheadAttention(embed_dim=256, num_heads=1)
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rnn_num_layers = rnn_num_layers



        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size,
                               num_layers=rnn_num_layers)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers)

        self.out = nn.Linear(rnn_hidden_size, output_size)

    def forward(self, scan_path, time_series, hidden):
        """
            Args:
                scan_path (torch.Tensor): must be 349x231 for now
                time_series (torch.Tensor):
            Returns:
                x (float): logit for confusion prediction - requires cross entropy loss
        """
        x1 = self.pool(F.relu(self.conv1(scan_path)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, self.n_channels_2 * self.mp2_out_hor * self.mp2_out_vert)
        x1 = F.relu(self.fc1(x1))

        # change input shape to (max_seq_size, batch_size, input_features):
        x2 = time_series.permute(1, 0, 2)
        x2, _ = self.multihead_attn1(x2, x2, x2, need_weights=False)
        x2, hidden = self.rnn(x2, hidden)
        #x2, _ = self.multihead_attn2(x2, x2, x2, need_weights=False)
        x2 = x2[-1, :, :]  # take only the last output

        x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def init_hidden(self, batch_size):
        """ Initializes the hidden state with zero tensors.
        """
        if self.rnn_type == 'lstm':
            return (autograd.Variable(torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size)).float().to(device),
                    autograd.Variable(torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size)).float().to(device))
        else:
            return autograd.Variable(torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size)).float().to(device)


def st_pickle_loader(input_file_path, max_length=870):
    """ Processes a raw data item into a scan path image and a time series
        for input into a STNet.

        Args:
            input_file_name (string): the name of the data item to be loaded
            max_length (int): max number of samples to use for a given item.
                If -1, use all samples
        Returns:
            item (numpy.ndarray): the fully processed data item for RNN input
            item_sp (PIL Image):

    """
    # Example of input_file_path = D:\Canary\multimodal-dl-framework
    # \dataset\alzheimer\tasks\cookie_theft\modalities\preprocessed\sequences
    # \eye_tracking\augmented\0_control\something.pkl

    transform = transforms.Compose([transforms.Resize((150,150)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    """
    0 GazePointLeftX (ADCSpx)
    1 GazePointLeftY (ADCSpx)
    2 GazePointRightX (ADCSpx)
    3 GazePointRightY (ADCSpx)
    4 GazePointX (ADCSpx)
    5 GazePointY (ADCSpx)
    6 GazePointX (MCSpx)
    7 GazePointY (MCSpx)
    8 GazePointLeftX (ADCSmm)
    9 GazePointLeftY (ADCSmm)
    10 GazePointRightX (ADCSmm)
    11 GazePointRightY (ADCSmm)
    12 DistanceLeft
    13 DistanceRight
    14 PupilLeft
    15 PupilRight
    16 FixationPointX (MCSpx)
    17 FixationPointY (MCSpx)
    18 FixationLength
    """

    file = open(input_file_path, 'rb')
    item = pickle.load(file)
    item = item.values
    item[:,0] = (item[:,0] + item[:,2])/2 #column 0 is now ave Gx
    item[:,1] = (item[:,1] + item[:,3])/2 #column 1 is now ave Gy
    item = item[item[:,0] > 0]
    item = abs(item)
    item[item == 1.0] = -1.0
    item = item[:,[0, 1, 12, 13, 14, 15]] # drop all but Gx average, Gy average, Left eye distance, left eye pupil, right eye distance, right eye pupil

    if len(item) == 0:
        item = np.zeros((1000, 6))
    else:
        if max_length != -1:
            item = item[-max_length:,:]
            if len(item) < max_length:
                num_zeros_to_pad = (max_length)-len(item)
                item = np.append(np.zeros((num_zeros_to_pad, len(item[0]))), item, axis=0)
    file.close()

    # input_filepath_example: r"D:\Canary\dataset\augmented\train2\patient\Gaze_HE-224-4.pkl"
    filename = input_file_path.split(os.sep)[-1].split('.')[0]
    category = input_file_path.split(os.sep)[-3]

    #check high low
    path_to_sp = os.path.join(BASE_DIR, "msnv_final_data_tasks", TASK, "high", "images", filename + '.png')

    # Check if the file exists, if not, try "low"
    if not os.path.exists(path_to_sp):
        path_to_sp = os.path.join(BASE_DIR, "msnv_final_data_tasks", TASK, "low", "images", filename + '.png')

    im = Image.open(path_to_sp)
    item_sp = transform(im)[0:3,:,:]
    return item, item_sp

"""# Cookie_theft"""

TASK = "Meara_label"

# Commented out IPython magic to ensure Python compatibility.
def cross_validate( model_type,
                    folds,
                    epochs,
                    criterion_type,
                    optimizer_type,
                    confused_path,
                    not_confused_path,
                    print_every,
                    plot_every,
                    hidden_size,
                    num_layers,
                    down_sample_training=False,
                    learning_rate=0.0001,
                    path_to_data_split = os.path.join(BASE_DIR, "meara_tasks.pickle"),
                    verbose=False,
                   patience=3
                  ):
    """
        Perform Cross Validation of the model using k-folds.

        Args:
            model_type (string): the type of RNN to use. Must be 'lstm', 'gru', or 'rnn'
            epochs (int): the max number of epochs to train the model for each fold
            criterion_type (string): the name loss function to use for training. Currently must be 'NLLLoss'
            optimizer_type (string): the name of learning algorithm to use for training. ex 'Adam'
            confused_path (string): the path to the folder containing the confused data samples
            not_confused_path (string): the path to the folder containing the not_confused data samples
            print_every (int): the number of batches to train for before printing relevant stats
            plot_every (int): the number of batches to train for before recording relevant stats, which
                will be plotted after each fold
            hidden_size (int): the number of hidden units for each layer of the RNN
            num_layers (int): the number of hidden_unit sized layers of the RNN
            down_sample_training (boolean): if True training set will be balanced by down sampling not_confused
            learning_rate (float): the first learning rate to be used by the optimizer
            path_to_data_split (string): relative path to the file containing the item names for each CV fold
            verbose (boolean): if True, function will print additional stats

        Returns: (list,list,list,list,list)
            cv_val_sens (list): list containing the validation sensitivity for each fold
            cv_val_spec (list): list containing the validation specificity for each fold
            cv_test_combined (list): list containing the combined test accuracy for each fold
    """

    #ensure same items appear in folds, for reproducibility:
    infile = open(path_to_data_split,'rb')
    split = pickle.load(infile)
    infile.close()

    train_confused_splits = split[0]
    test_confused_splits = split[1]
    train_not_confused_splits = split[2]
    test_not_confused_splits = split[3]


    cv_test_sens = []
    cv_test_spec = []
    cv_test_combined = []
    cv_auc = []

    for k in range(folds):
        print("\nFold ", k+1)
        # Get data item file names for this fold and downsample not_confused to balance training set
        train_confused, \
        train_not_confused, \
        val_confused, \
        val_not_confused = \
        utils.get_train_val_split(train_confused_splits[k],
                                  train_not_confused_splits[k],
                                  percent_val_set=0.2)

        if down_sample_training:
            if len(train_not_confused) > len(train_confused):
                train_not_confused = random.sample(train_not_confused, k=(len(train_confused)))
            elif len(train_confused) > len(train_not_confused):
                train_confused = random.sample(train_confused, k=(len(train_not_confused)))

        test_confused = test_confused_splits[k]
        test_not_confused = test_not_confused_splits[k]

        if verbose:
            print("Patient items in training set: ", len(train_confused))
            print("Control items in training set: ", len(train_not_confused))
            print("Patient items in validation set: ", len(val_confused))
            print("Control items in validation set: ", len(val_not_confused))

        if verbose:
            print("\nTest patient items:\n")
            print(test_confused)

        local_train_confused_path = os.path.join(BASE_DIR, 'dataset/augmented_meara/train_cookie_theft/patient/')
        local_val_confused_path = os.path.join(BASE_DIR, 'dataset/augmented_meara/val_cookie_theft/patient/')
        local_test_confused_path = os.path.join(BASE_DIR, 'dataset/augmented_meara/test_cookie_theft/patient/')
        local_train_not_confused_path = os.path.join(BASE_DIR, 'dataset/augmented_meara/train_cookie_theft/control/')
        local_val_not_confused_path = os.path.join(BASE_DIR, 'dataset/augmented_meara/val_cookie_theft/control/')
        local_test_not_confused_path = os.path.join(BASE_DIR, 'dataset/augmented_meara/test_cookie_theft/control/')

        # Remove any old directories
        if os.path.exists(local_train_confused_path):
            shutil.rmtree(local_train_confused_path)
        if os.path.exists(local_val_confused_path):
            shutil.rmtree(local_val_confused_path)
        if os.path.exists(local_test_confused_path):
            shutil.rmtree(local_test_confused_path)

        if os.path.exists(local_train_not_confused_path):
            shutil.rmtree(local_train_not_confused_path)
        if os.path.exists(local_val_not_confused_path):
            shutil.rmtree(local_val_not_confused_path)
        if os.path.exists(local_test_not_confused_path):
            shutil.rmtree(local_test_not_confused_path)

        # Make new temp directories
        os.makedirs(local_train_confused_path)
        for i in train_confused:
            shutil.copy(src=confused_path+i,dst=local_train_confused_path+i)

        os.makedirs(local_val_confused_path)
        for i in val_confused:
            shutil.copy(src=confused_path+i,dst=local_val_confused_path+i)

        os.makedirs(local_test_confused_path)
        for i in test_confused:
            shutil.copy(src=confused_path+i,dst=local_test_confused_path+i)

        os.makedirs(local_train_not_confused_path)
        for i in train_not_confused:
            shutil.copy(src=not_confused_path+i,dst=local_train_not_confused_path+i)

        os.makedirs(local_val_not_confused_path)
        for i in val_not_confused:
            shutil.copy(src=not_confused_path+i,dst=local_val_not_confused_path+i)

        os.makedirs(local_test_not_confused_path)
        for i in test_not_confused:
            shutil.copy(src=not_confused_path+i,dst=local_test_not_confused_path+i)

        # Prepare training and validation data
        trainset = datasets.DatasetFolder(os.path.join(BASE_DIR, 'dataset/augmented_meara/train_cookie_theft'),
                                               loader=st_pickle_loader,
                                               extensions='.pkl')

        valset = datasets.DatasetFolder(os.path.join(BASE_DIR, 'dataset/augmented_meara/val_cookie_theft'),
                                                 loader=st_pickle_loader,
                                                 extensions='.pkl')

        testset = datasets.DatasetFolder(os.path.join(BASE_DIR, 'dataset/augmented_meara/test_cookie_theft'),
                                                 loader=st_pickle_loader,
                                                 extensions='.pkl')


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, drop_last=True)


        valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, drop_last=True)

        print("Training data: ", trainset)
        print("Validation data: ", valset)
        print("Test data: ", testset)

        torch.manual_seed(MANUAL_SEED)
        if model_type == 'gru':
            model = VTNet(rnn_type='gru', rnn_num_layers=num_layers, rnn_hidden_size=hidden_size).float().to(device)
        elif model_type == 'lstm':
            model = VTNet(rnn_type='lstm', rnn_num_layers=num_layers, rnn_hidden_size=hidden_size).float().to(device)
        else:
            model = VTNet(rnn_type='rnn', rnn_num_layers=num_layers, rnn_hidden_size=hidden_size).float().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #save fresh model to clear any old ones out
        torch.save(model.state_dict(), './best_STNet_fold_pd_'+str(k) +'.pt')
        best_val_combined = 0.0
        #Train model
        epochs_without_improvement = 0
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                item, item_sp = inputs[0].float(), inputs[1].float()
                item, item_sp, labels = item.to(device), item_sp[:,0,:,:].unsqueeze(1).to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                hidden = model.init_hidden(BATCH_SIZE)
                # forward + backward + optimize
                outputs = model(scan_path=item_sp, time_series=item, hidden=hidden)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 10 == 0:
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
            print('epoch %d average training loss: %.5f' % (epoch + 1, epoch_loss/ len(trainloader)))

            #check validation set metrics
            running_val_loss = 0.0
            y_true = torch.zeros((len(valloader)*4))
            y_scores = torch.zeros((len(valloader)*4, 2))
            with torch.no_grad():

                for i, data in enumerate(valloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    item, item_sp = inputs[0].float(), inputs[1].float()
                    item, item_sp, labels = item.to(device), item_sp[:,0,:,:].unsqueeze(1).to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    hidden = model.init_hidden(BATCH_SIZE)
                    # forward + backward + optimize
                    outputs = model(scan_path=item_sp, time_series=item, hidden=hidden)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    #keep track of predictions
                    num_items = outputs.shape[0]
                    y_true[i*num_items: i*num_items + num_items] = labels
                    y_scores[i*num_items: i*num_items + num_items, :] = outputs.squeeze()

                val_loss = running_val_loss/ len(valloader)
                print('epoch %d average val loss: %.5f' % (epoch + 1, val_loss))

                #check metrics:
                # no option to specify positive label, so flipping for confused=1
                y_true_flipped = np.array(y_true.numpy(), copy=True)
                y_true_flipped[y_true == 1] = 0
                y_true_flipped[y_true == 0] = 1
                #auc = roc_auc_score(y_true_flipped, y_scores.numpy()[:,0])
                # roc_curve expects y_scores to be probability values of the positive class
                fpr, tpr, thresholds = roc_curve(y_true, y_scores.numpy()[:,0], pos_label=0)

                sensitivity, specificity, \
                accuracy = utils.optimal_threshold_sensitivity_specificity(thresholds[1:],
                                                                           tpr[1:],
                                                                           fpr[1:],
                                                                           y_true,
                                                                           y_scores.numpy()[:,0])
                combined = (sensitivity + specificity ) / 2.0
                print("epoch %d validation sens. : %.5f, spec. : %.5f; combined: %.5f" % (epoch + 1, sensitivity, specificity, combined))
                if combined > best_val_combined:
                    print("New best validation combined accuracy found. Saving model...")
                    best_val_combined = combined
                    torch.save(model.state_dict(), './best_base_STNet_fold_pd_'+str(k) +'.pt')
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement = epochs_without_improvement + 1

                print("\n Epochs without improvement = ", epochs_without_improvement)

            if epochs_without_improvement == patience:
                print("\n Stopped training because {} epochs without improvement. . .".format(patience))
                break

        y_true = torch.tensor([]).to(device)
        y_scores = torch.tensor([]).to(device)
        with torch.no_grad():
            model.load_state_dict(torch.load('./best_base_STNet_fold_pd_'+str(k) +'.pt', map_location=device))
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                item, item_sp = inputs[0].float(), inputs[1].float()
                item, item_sp, labels = item.to(device), item_sp[:,0,:,:].unsqueeze(1).to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                hidden = model.init_hidden(BATCH_SIZE)
                # forward + backward + optimize
                outputs = model(scan_path=item_sp, time_series=item, hidden=hidden)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                #keep track of predictions
                num_items = outputs.shape[0]
                y_true = torch.cat((y_true, labels))
                y_scores = torch.cat((y_scores,outputs.squeeze()))
                #y_true[i*num_items: i*num_items + num_items] = labels
                #y_scores[i*num_items: i*num_items + num_items, :] = outputs.squeeze()

            #y_pred = torch.argmax(y_scores, axis=1)

            #check metrics:
            # no option to specify positive label, so flipping for confused=1
            y_true = y_true.cpu()
            y_scores = y_scores.cpu()
            y_true_flipped = np.array(y_true.numpy(), copy=True)
            y_true_flipped[y_true == 1] = 0
            y_true_flipped[y_true == 0] = 1
            auc = roc_auc_score(y_true_flipped, y_scores.numpy()[:,0])
            # roc_curve expects y_scores to be probability values of the positive class
            fpr, tpr, thresholds = roc_curve(y_true, y_scores.numpy()[:,0], pos_label=0)

            sensitivity, specificity, \
            accuracy = utils.optimal_threshold_sensitivity_specificity(thresholds[1:],
                                                                       tpr[1:],
                                                                       fpr[1:],
                                                                       y_true,
                                                                       y_scores.numpy()[:,0])
            combined = (sensitivity + specificity ) / 2.0

            print("Test set sens. : %.5f, spec. : %.5f, combined: %.5f, auc: %.5f" % (sensitivity, specificity, combined, auc))
            cv_test_sens.append(sensitivity)
            cv_test_spec.append(specificity)
            cv_test_combined.append(combined)
            cv_auc.append(auc)

    print("\n Average 10-fold CV test sensitivity: %.5f, specificity: %.5f, combined: %.5f, AUC: %.5f" %
          ((sum(cv_test_sens)/len(cv_test_sens)),
           (sum(cv_test_spec)/len(cv_test_spec)),
           (sum(cv_test_combined)/len(cv_test_combined)),
           (sum(cv_auc)/len(cv_auc))))
    return cv_test_sens, cv_test_spec, cv_test_combined, cv_auc

# compute for 10 different seeds
sens = []
spec = []
comb = []
auc = []


for i in range(1):
# baseline 10-fold CV with GRU
    np.random.seed(MANUAL_SEED+i)
    random.seed(MANUAL_SEED+i)
    torch.manual_seed(MANUAL_SEED+i)

    sens_list, spec_list, comb_list, auc_list = cross_validate(model_type='gru',
                                               folds=10,
                                               epochs=100,
                                               criterion_type='NLLLoss',
                                               optimizer_type='Adam',
                                               confused_path=os.path.join(
    BASE_DIR, "msnv_final_data_tasks",
    TASK, "high/pickle_files/"
),
                                               not_confused_path=os.path.join(
    BASE_DIR, "msnv_final_data_tasks",
    TASK, "low/pickle_files/"
),
                                               print_every=1,
                                               plot_every=1,
                                               hidden_size=HIDDEN_SIZE,
                                               down_sample_training=False,
                                               num_layers=1,
                                               learning_rate=0.0001,
                                               verbose=True)
    # add mean of each measure for 10-fold CV to list
    sens.append(np.mean(sens_list))
    spec.append(np.mean(spec_list))
    comb.append(np.mean(comb_list))
    auc.append(np.mean(auc_list))

print("sensitivities: ", sens)
print("specificities: ", spec)
print("combined: ", comb)
print("auc: ", auc)

print("average sensitivity: ", np.mean(sens))
print("average specificity: ", np.mean(spec))
print("average combined: ", np.mean(comb))
print("average auc: ", np.mean(auc))

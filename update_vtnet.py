BASE_DIR = "/ubc/cs/research/ubc_ml/ewong25"

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
from torch.utils.checkpoint import checkpoint

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

BATCH_SIZE = 4
MANUAL_SEED = 1
HIDDEN_SIZE = 256
INPUT_SIZE = 6
utils.INPUT_SIZE = INPUT_SIZE

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
                 img_n_hor=150,
                 attn_chunk_size=1024,    # chunk size for first attention block
                 tbptt_step=512           # segment length for TBPTT in the RNN
                 ):
        """
        Args:
            input_size (int): number of features per time step.
            rnn_hidden_size (int): number of hidden units for the RNN.
            output_size (int): number of output classes.
            batch_size (int): batch size.
            rnn_type (str): type of RNN to use ('gru', 'lstm', or 'rnn').
            rnn_num_layers (int): number of layers in the RNN.
            n_channels_1 (int): number of channels in first conv layer.
            kernel_size_1 (int): kernel size for first conv layer.
            n_channels_2 (int): number of channels in second conv layer.
            kernel_size_2 (int): kernel size for second conv layer.
            img_n_vert (int): vertical dimension of the image.
            img_n_hor (int): horizontal dimension of the image.
            attn_chunk_size (int): maximum sequence length processed at once in attention.
            tbptt_step (int): step size for truncated backpropagation through time.
        """
        super(VTNet, self).__init__()
        self.n_channels_2 = n_channels_2
        self.attn_chunk_size = attn_chunk_size
        self.tbptt_step = tbptt_step

        # CNN portion
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels_1, kernel_size=kernel_size_1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n_channels_1, out_channels=n_channels_2, kernel_size=kernel_size_2, stride=1)
        # output size calculations
        self.conv1_out_vert = img_n_vert - kernel_size_1 + 1
        self.conv1_out_hor = img_n_hor - kernel_size_1 + 1
        self.mp1_out_vert = int(np.floor((self.conv1_out_vert - 2) / 2) + 1)
        self.mp1_out_hor = int(np.floor((self.conv1_out_hor - 2) / 2) + 1)
        self.conv2_out_vert = self.mp1_out_vert - kernel_size_2 + 1
        self.conv2_out_hor = self.mp1_out_hor - kernel_size_2 + 1
        self.mp2_out_vert = int(np.floor((self.conv2_out_vert - 2) / 2) + 1)
        self.mp2_out_hor = int(np.floor((self.conv2_out_hor - 2) / 2) + 1)
        self.fc1 = nn.Linear(n_channels_2 * self.mp2_out_hor * self.mp2_out_vert, 50)
        self.fc2 = nn.Linear(rnn_hidden_size + 50, 20)
        self.fc3 = nn.Linear(20, output_size)

        # RNN and attention portion
        # For the time-series branch we use multihead attention and an RNN.
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=6, num_heads=1)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=1)
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rnn_num_layers = rnn_num_layers

        # Layer normalization to stabilize training
        self.layernorm1 = nn.LayerNorm(6)
        self.layernorm2 = nn.LayerNorm(rnn_hidden_size)

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

    def chunked_attention(self, x, attn_module, chunk_size):
        """
        Splits sequence x into chunks and applies the attention module on each chunk.
        Args:
            x (Tensor): shape (seq_length, batch_size, embed_dim)
            attn_module: attention module (e.g., multihead_attn1)
            chunk_size (int): maximum number of timesteps per chunk.
        Returns:
            Tensor with shape (seq_length, batch_size, embed_dim) after attention.
        """
        seq_length, batch_size, embed_dim = x.size()
        outputs = []
        for i in range(0, seq_length, chunk_size):
            x_chunk = x[i: i + chunk_size]  # (chunk, batch_size, embed_dim)
            # Apply attention on the chunk.
            attn_out, _ = attn_module(x_chunk, x_chunk, x_chunk, need_weights=False)
            outputs.append(attn_out)
        return torch.cat(outputs, dim=0)

    def tbptt_forward(self, x_chunk, hidden):
        """
        A helper function for the RNN call to enable gradient checkpointing.
        """
        return self.rnn(x_chunk, hidden)

    def forward(self, scan_path, time_series, hidden):
        """
        Args:
            scan_path (Tensor): image input.
            time_series (Tensor): time-series data.
            hidden: initial hidden state for the RNN.
        Returns:
            x: final output logits.
        """
        # Process image through CNN
        x1 = self.pool(F.relu(self.conv1(scan_path)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, self.n_channels_2 * self.mp2_out_hor * self.mp2_out_vert)
        x1 = F.relu(self.fc1(x1))

        # Process time-series data:
        # Permute to (seq_length, batch_size, input_features)
        x2 = time_series.permute(1, 0, 2)
        # Apply chunked attention on the long sequence and then layer normalize.
        x2 = self.chunked_attention(x2, self.multihead_attn1, self.attn_chunk_size)
        x2 = self.layernorm1(x2)

        # TBPTT: Process x2 in segments through the RNN with gradient checkpointing.
        rnn_outputs = []
        seq_length = x2.size(0)
        for i in range(0, seq_length, self.tbptt_step):
            x_chunk = x2[i: i + self.tbptt_step]
            # Use checkpointing to save memory.
            x_chunk, hidden = checkpoint(self.tbptt_forward, x_chunk, hidden)
            # Detach hidden state to truncate gradients
            if self.rnn_type == 'lstm':
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()
            rnn_outputs.append(x_chunk)
        x2 = torch.cat(rnn_outputs, dim=0)
        x2 = self.layernorm2(x2)

        # Apply a second full attention layer on the RNN outputs.
        x2, _ = self.multihead_attn2(x2, x2, x2, need_weights=False)
        # Take the output of the last time step.
        x2 = x2[-1, :, :]

        # Combine features from CNN and time-series branches.
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_hidden(self, batch_size):
        """Initializes the hidden state with zeros."""
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device))
        else:
            return torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)

def st_pickle_loader(input_file_path, max_length=35623):
    """Processes a raw data item into a scan path image and a time series for input into VTNet."""
    transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    ])

    file = open(input_file_path, 'rb')
    item = pickle.load(file)
    item = item.values
    item[:,0] = (item[:,0] + item[:,2]) / 2  # average Gaze X
    item[:,1] = (item[:,1] + item[:,3]) / 2  # average Gaze Y
    item = item[item[:,0] > 0]
    item = abs(item)
    item[item == 1.0] = -1.0
    item = item[:, [0, 1, 12, 13, 14, 15]]  # select relevant columns

    if len(item) == 0:
        item = np.zeros((max_length, 6))
    else:
        if max_length != -1:
            item = item[-max_length:, :]
            if len(item) < max_length:
                num_zeros_to_pad = max_length - len(item)
                item = np.append(np.zeros((num_zeros_to_pad, item.shape[1])), item, axis=0)
    file.close()

    filename = input_file_path.split(os.sep)[-1].split('.')[0]
    category = input_file_path.split(os.sep)[-3]

    # Determine high/low directory for the image.
    path_to_sp = os.path.join(BASE_DIR, "msnv_final_data_combined", TASK, "high", "images", filename + '.png')
    if not os.path.exists(path_to_sp):
        path_to_sp = os.path.join(BASE_DIR, "msnv_final_data_combined", TASK, "low", "images", filename + '.png')

    im = Image.open(path_to_sp)
    item_sp = transform(im)[0:3, :, :]
    return item, item_sp

TASK = "BarChartLit_label"

def cross_validate(model_type,
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
                   path_to_data_split=os.path.join(BASE_DIR, "barchart_lit_combined.pickle"),
                   verbose=False,
                   patience=3):
    """
    Perform Cross Validation using k-folds.
    """
    infile = open(path_to_data_split, 'rb')
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
        train_confused, train_not_confused, val_confused, val_not_confused = utils.get_train_val_split(
            train_confused_splits[k], train_not_confused_splits[k], percent_val_set=0.2)

        if down_sample_training:
            if len(train_not_confused) > len(train_confused):
                train_not_confused = random.sample(train_not_confused, k=len(train_confused))
            elif len(train_confused) > len(train_not_confused):
                train_confused = random.sample(train_confused, k=len(train_not_confused))

        test_confused = test_confused_splits[k]
        test_not_confused = test_not_confused_splits[k]

        if verbose:
            print("Patient items in training set: ", len(train_confused))
            print("Control items in training set: ", len(train_not_confused))
            print("Patient items in validation set: ", len(val_confused))
            print("Control items in validation set: ", len(val_not_confused))
            print("\nTest patient items:\n", test_confused)

        local_train_confused_path = os.path.join(BASE_DIR, 'dataset_across/augmented/train_cookie_theft/patient/')
        local_val_confused_path = os.path.join(BASE_DIR, 'dataset_across/augmented/val_cookie_theft/patient/')
        local_test_confused_path = os.path.join(BASE_DIR, 'dataset_across/augmented/test_cookie_theft/patient/')
        local_train_not_confused_path = os.path.join(BASE_DIR, 'dataset_across/augmented/train_cookie_theft/control/')
        local_val_not_confused_path = os.path.join(BASE_DIR, 'dataset_across/augmented/val_cookie_theft/control/')
        local_test_not_confused_path = os.path.join(BASE_DIR, 'dataset_across/augmented/test_cookie_theft/control/')

        # Remove any old directories and recreate.
        for path in [local_train_confused_path, local_val_confused_path, local_test_confused_path,
                     local_train_not_confused_path, local_val_not_confused_path, local_test_not_confused_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        for i in train_confused:
            shutil.copy(src=confused_path + i, dst=local_train_confused_path + i)
        for i in val_confused:
            shutil.copy(src=confused_path + i, dst=local_val_confused_path + i)
        for i in test_confused:
            shutil.copy(src=confused_path + i, dst=local_test_confused_path + i)
        for i in train_not_confused:
            shutil.copy(src=not_confused_path + i, dst=local_train_not_confused_path + i)
        for i in val_not_confused:
            shutil.copy(src=not_confused_path + i, dst=local_val_not_confused_path + i)
        for i in test_not_confused:
            shutil.copy(src=not_confused_path + i, dst=local_test_not_confused_path + i)

        trainset = datasets.DatasetFolder(os.path.join(BASE_DIR, 'dataset_across/augmented/train_cookie_theft'),
                                          loader=st_pickle_loader,
                                          extensions='.pkl')
        valset = datasets.DatasetFolder(os.path.join(BASE_DIR, 'dataset_across/augmented/val_cookie_theft'),
                                        loader=st_pickle_loader,
                                        extensions='.pkl')
        testset = datasets.DatasetFolder(os.path.join(BASE_DIR, 'dataset_across/augmented/test_cookie_theft'),
                                         loader=st_pickle_loader,
                                         extensions='.pkl')

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, drop_last=True)

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

        # Save fresh model checkpoint.
        torch.save(model.state_dict(), './best_STNet_fold_pd_' + str(k) + '.pt')
        best_val_combined = 0.0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                item, item_sp = inputs[0].float(), inputs[1].float()
                item = item.to(device)
                item_sp = item_sp[:, 0, :, :].unsqueeze(1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                hidden = model.init_hidden(BATCH_SIZE)
                outputs = model(scan_path=item_sp, time_series=item, hidden=hidden)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 10 == 0:
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
            print('epoch %d average training loss: %.5f' % (epoch + 1, epoch_loss / len(trainloader)))

            # Validation loop.
            running_val_loss = 0.0
            y_true = torch.zeros((len(valloader) * 4))
            y_scores = torch.zeros((len(valloader) * 4, 2))
            with torch.no_grad():
                for i, data in enumerate(valloader, 0):
                    inputs, labels = data
                    item, item_sp = inputs[0].float(), inputs[1].float()
                    item = item.to(device)
                    item_sp = item_sp[:, 0, :, :].unsqueeze(1).to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    hidden = model.init_hidden(BATCH_SIZE)
                    outputs = model(scan_path=item_sp, time_series=item, hidden=hidden)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                    num_items = outputs.shape[0]
                    y_true[i * num_items: i * num_items + num_items] = labels
                    y_scores[i * num_items: i * num_items + num_items, :] = outputs.squeeze()

                val_loss = running_val_loss / len(valloader)
                print('epoch %d average val loss: %.5f' % (epoch + 1, val_loss))

                y_true_flipped = np.array(y_true.numpy(), copy=True)
                y_true_flipped[y_true == 1] = 0
                y_true_flipped[y_true == 0] = 1
                fpr, tpr, thresholds = roc_curve(y_true, y_scores.numpy()[:, 0], pos_label=0)
                sensitivity, specificity, accuracy = utils.optimal_threshold_sensitivity_specificity(
                    thresholds[1:], tpr[1:], fpr[1:], y_true, y_scores.numpy()[:, 0])
                combined = (sensitivity + specificity) / 2.0
                print("epoch %d validation sens. : %.5f, spec. : %.5f; combined: %.5f" %
                      (epoch + 1, sensitivity, specificity, combined))
                if combined > best_val_combined:
                    print("New best validation combined accuracy found. Saving model...")
                    best_val_combined = combined
                    torch.save(model.state_dict(), './best_base_STNet_fold_pd_' + str(k) + '.pt')
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                print("\n Epochs without improvement = ", epochs_without_improvement)

            if epochs_without_improvement == patience:
                print("\n Stopped training because {} epochs without improvement. . .".format(patience))
                break

        y_true = torch.tensor([]).to(device)
        y_scores = torch.tensor([]).to(device)
        with torch.no_grad():
            model.load_state_dict(torch.load('./best_base_STNet_fold_pd_' + str(k) + '.pt', map_location=device))
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                item, item_sp = inputs[0].float(), inputs[1].float()
                item = item.to(device)
                item_sp = item_sp[:, 0, :, :].unsqueeze(1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                hidden = model.init_hidden(BATCH_SIZE)
                outputs = model(scan_path=item_sp, time_series=item, hidden=hidden)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                num_items = outputs.shape[0]
                y_true = torch.cat((y_true, labels))
                y_scores = torch.cat((y_scores, outputs.squeeze()))

            y_true = y_true.cpu()
            y_scores = y_scores.cpu()
            y_true_flipped = np.array(y_true.numpy(), copy=True)
            y_true_flipped[y_true == 1] = 0
            y_true_flipped[y_true == 0] = 1
            auc = roc_auc_score(y_true_flipped, y_scores.numpy()[:, 0])
            fpr, tpr, thresholds = roc_curve(y_true, y_scores.numpy()[:, 0], pos_label=0)
            sensitivity, specificity, accuracy = utils.optimal_threshold_sensitivity_specificity(
                thresholds[1:], tpr[1:], fpr[1:], y_true, y_scores.numpy()[:, 0])
            combined = (sensitivity + specificity) / 2.0
            print("Test set sens. : %.5f, spec. : %.5f, combined: %.5f, auc: %.5f" %
                  (sensitivity, specificity, combined, auc))
            cv_test_sens.append(sensitivity)
            cv_test_spec.append(specificity)
            cv_test_combined.append(combined)
            cv_auc.append(auc)

    print("\n Average 10-fold CV test sensitivity: %.5f, specificity: %.5f, combined: %.5f, AUC: %.5f" %
          ((sum(cv_test_sens) / len(cv_test_sens)),
           (sum(cv_test_spec) / len(cv_test_spec)),
           (sum(cv_test_combined) / len(cv_test_combined)),
           (sum(cv_auc) / len(cv_auc))))
    return cv_test_sens, cv_test_spec, cv_test_combined, cv_auc

sens = []
spec = []
comb = []
auc = []

for i in range(1):
    # baseline 10-fold CV with GRU
    np.random.seed(MANUAL_SEED + i)
    random.seed(MANUAL_SEED + i)
    torch.manual_seed(MANUAL_SEED + i)

    sens_list, spec_list, comb_list, auc_list = cross_validate(
        model_type='gru',
        folds=10,
        epochs=100,
        criterion_type='NLLLoss',
        optimizer_type='Adam',
        confused_path=os.path.join(BASE_DIR, "msnv_final_data_combined", TASK, "high/pickle_files/"),
        not_confused_path=os.path.join(BASE_DIR, "msnv_final_data_combined", TASK, "low/pickle_files/"),
        print_every=1,
        plot_every=1,
        hidden_size=HIDDEN_SIZE,
        down_sample_training=False,
        num_layers=1,
        learning_rate=0.0001,
        verbose=True
    )
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

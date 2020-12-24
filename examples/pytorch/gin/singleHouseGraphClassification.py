'''

This code is responsible for training graph model with input data as single house, with leave day out for testing approach.

This runs for all configuration[Raw, OB_Compressed, OB_decompressed].

And, it runs for all the 5 houses.

It creates the graph of the house if that does not exist from csv and nodes, and edge data present. Else, it uses graph data
already present.

'''
import random
import sys
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import os

from pytorchtools import EarlyStopping
from sklearn.model_selection import LeaveOneOut
from dataloader import GraphDataLoader, collate
from parser import Parser
from gin import GIN
import pandas as pd
import datetime
from datetime import datetime
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl

from dgl.data.utils import save_graphs, load_graphs

config = {
    "batch_size": 32,
    "ActivityIdList":
         [{'name': 'washDishes', 'id': 0},
         {'name': 'goToBed', 'id': 1},
         {'name': 'brushTeeth', 'id': 2},
         {'name': 'prepareLunch', 'id': 3},
         {'name': 'eating', 'id': 4},
         {'name': 'takeShower', 'id': 5},
         {'name': 'leaveHouse', 'id': 6},
         {'name': 'getDrink', 'id': 7},
         {'name': 'prepareBreakfast', 'id': 8},
         {'name': 'getSnack', 'id': 9},
         {'name': 'idle', 'id': 10},
         {'name': 'grooming', 'id': 11},
         {'name': 'prepareDinner', 'id': 12},
         {'name': 'relaxing', 'id': 13},
         {'name': 'useToilet', 'id': 14}],
"merging_activties" : {
        "loadDishwasher": "washDishes",
        "unloadDishwasher": "washDishes",
        "loadWashingmachine": "washClothes",
        "unloadWashingmachine": "washClothes",
        "receiveGuest": "relaxing",
        "eatDinner": "eating",
        "eatBreakfast": "eating",
        "getDressed": "grooming",
        "shave": "grooming",
        "takeMedication": "idle",
        "leave_Home": "leaveHouse",
        "Sleeping": "goToBed",
        "Bed_to_Toilet": "useToilet",
        "Enter_Home": "idle",
        "Respirate": "relaxing",
        "Work": "idle",
        "Housekeeping": "idle",
        "Idle": "idle",
        "watchTV": "relaxing"
    },
}

def getClassnameFromID(train_label):

    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["id"] == int(train_label)]
    return train_label[0]['name']

def train(args, net, trainloader, optimizer, criterion, epoch, decompressed_csv_path = None, house = None, run_configuration=None):
    net.train()
    running_loss = 0
    total_iters = len(trainloader)

    for (graphs, labels) in trainloader:
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr').to(args.device)
        graphs = graphs.to(args.device)
        outputs, _ = net(graphs, feat)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss = running_loss / total_iters
    return running_loss

def eval_net(args, net, dataloader, criterion, text = 'train'):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0
    f1 = 0
    all_labels = []
    all_predicted = []
    nb_classes = args.nb_classes
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    hiddenLayerEmbeddings = None

    for data in dataloader:
        graphs, labels = data
        feat = graphs.ndata.pop('attr').to(args.device)
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        total += len(labels)
        outputs, hiddenLayerEmbeddings = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

        all_labels.extend(labels.cpu())
        all_predicted.extend(predicted.cpu())

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_acc = per_class_acc.cpu().numpy()
    per_class_acc[np.isnan(per_class_acc)] = -1
    per_class_acc_dict = {}
    for i, entry in enumerate(per_class_acc):
        if entry != -1:
            per_class_acc_dict[getClassnameFromID(i)] = entry

    f1 = f1_score(all_labels, all_predicted, average='macro')


    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc, f1, per_class_acc_dict, confusion_matrix

def getIDFromClassName(train_label, config):
    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["name"] == train_label]
    return train_label[0]['id']

# Dataset Class
class GraphHouseDataset():
    def __init__(self, graphs, labels):
        super(GraphHouseDataset, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def __getitem__(self, idx):
        """ Get graph and label by index"""
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)


def _split_rand(labels, split_ratio=0.8, seed=0, shuffle=True):
    num_entries = len(labels)
    indices = list(range(num_entries))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(np.math.floor(split_ratio * num_entries))
    train_idx, valid_idx = indices[:split], indices[split:]

    print(
        "train_set : test_set = %d : %d",
        len(train_idx), len(valid_idx))

    return train_idx, valid_idx

def getStartAndEndIndex(df, test_index):
    # this line converts the string object in Timestamp object
    date = df['start'].iloc[test_index].item()
    index = df.index[df['start'] == date].tolist()
    # get start and end of this date
    return index[0], index[-1]

def getUniqueStartIndex(df):
    # this line converts the string object in Timestamp object
    if isinstance(df['start'][0], str):
        df['start'] = [datetime.strptime(d, '%d-%b-%Y %H:%M:%S') for d in df["start"]]
    # extracting date from timestamp
    if isinstance(df['start'][0], datetime):
        df['start'] = [datetime.date(d) for d in df['start']]
    s = df['start']
    return s[s.diff().dt.days != 0].index.values

def main(args, shuffle=True, decompressed_csv_path=None, ob_csv_file_path=None):
    file_names = ['ordonezB', 'houseB', 'houseC', 'houseA', 'ordonezA']

    # run_time_configs = ['ob_data_compressed', 'raw_data', 'ob_data_Decompressed']
    run_time_configs = ['raw_data']
    for run_configuration in run_time_configs:
        results_list = []
        print('\n\n\n\n Running configuration', run_configuration, '\n\n\n\n')
        for file_name in file_names:
            print('house is: ', file_name)
            if run_configuration is 'raw_data':
                config['ob_data_compressed'] = False
                config['ob_data_Decompressed'] = False
                config['raw_data'] = True

            elif run_configuration is 'ob_data_compressed':
                config['ob_data_compressed'] = True
                config['ob_data_Decompressed'] = False
                config['raw_data'] = False

            elif run_configuration is 'ob_data_Decompressed':
                config['ob_data_compressed'] = False
                config['ob_data_Decompressed'] = True
                config['raw_data'] = False

            if config['ob_data_compressed']:
                ob_csv_file_path = os.path.join(os.getcwd(), '../../../', 'data', file_name, 'ob_' + file_name + '.csv')
                decompressed_csv_path = os.path.join(os.getcwd(), '../../../', 'data', file_name, 'ob_' + file_name + '.csv')

            elif config['raw_data']:
                ob_csv_file_path = os.path.join(os.getcwd(), '../../../', 'data', file_name, file_name + '.csv')
                decompressed_csv_path = os.path.join(os.getcwd(), '../../../', 'data', file_name, file_name + '.csv')

            elif config['ob_data_Decompressed']:
                ob_csv_file_path = os.path.join(os.getcwd(), '../../../', 'data', file_name, 'ob_' + file_name + '.csv')
                decompressed_csv_path = os.path.join(os.getcwd(), '../../../', 'data', file_name, 'ob_decompressed_' + file_name + '.csv')

            # # set up seeds, args.seed supported
            # torch.manual_seed(seed=args.seed)
            # np.random.seed(seed=args.seed)

            is_cuda = not args.disable_cuda and torch.cuda.is_available()
            is_cuda = False

            if is_cuda:
                args.device = torch.device("cuda:" + str(args.device))
                torch.cuda.manual_seed_all(seed=args.seed)
            else:
                args.device = torch.device("cpu")

            if config['raw_data']:
                graph_path = os.path.join('../../../data', file_name, file_name + '.bin')
                # graph_path = os.path.join('../../../data/all_houses/all_houses_raw.bin')
            elif config['ob_data_compressed']:
                graph_path = os.path.join('../../../data', file_name,  'ob_' + file_name + '.bin')
            elif config['ob_data_Decompressed']:
                decompressedGraphPath = os.path.join('../../../data', file_name, file_name + '.bin')
                graph_path = os.path.join('../../../data', file_name, 'ob_' + file_name + '.bin')

            graphs = []
            labels = []

            if not os.path.exists(graph_path):
                print('\n\n\n\n')
                print('*******************************************************************')
                print('\t\t\t\t\t' + file_name + '\t\t\t\t\t\t\t')
                print('*******************************************************************')
                print('\n\n\n\n')

                nodes = pd.read_csv('../../../data/' + file_name + '/nodes.csv')
                edges = pd.read_csv('../../../data/' + file_name + '/bidrectional_edges.csv')

                if config['ob_data_compressed']:
                    house = pd.read_csv('../../../data/' + file_name + '/ob_' + file_name + '.csv')
                    lastChangeTimeInMinutes = pd.read_csv('../../../data/' + file_name + '/' + 'ob-house' + '-sensorChangeTime.csv')
                elif config['raw_data']:
                    house = pd.read_csv('../../../data/' + file_name + '/' + file_name + '.csv')
                    lastChangeTimeInMinutes = pd.read_csv('../../../data/' + file_name + '/' + 'house' + '-sensorChangeTime.csv')

                u = edges['Src']
                v = edges['Dst']

                # Create Graph per row of the House CSV

                # Combine Feature like this: Value, Place_in_House, Type, Last_change_Time_in_Second for each node
                for i in range(len(house)):
                # for i in range(5000):
                    feature = []
                    flag = 0
                    prev_node_value = 0
                    prev_node_change_time = 0
                    # Define Graph
                    g = dgl.graph((u, v))
                    node_num = 0
                    total_nodes = len(nodes)
                    # Add Features
                    for j in range(total_nodes - 1):
                        if nodes.loc[j, 'Type'] == 1:
                            node_value = -1
                            node_place_in_house = nodes.loc[j, 'place_in_house']
                            node_type = nodes.loc[j, 'Type']
                            feature.append([node_value, node_place_in_house, node_type, -1])
                            node_num += 1
                            continue

                        if flag == 0:
                            node_value = house.iloc[i, 4 + j - node_num]
                            last_change_time_in_minutes = lastChangeTimeInMinutes.iloc[i, 4 + j - node_num]
                            node_place_in_house = nodes.loc[j, 'place_in_house']
                            node_type = nodes.loc[j, 'Type']
                            feature.append([node_value, node_place_in_house, node_type, last_change_time_in_minutes])
                            if nodes.loc[j, 'Object'] == nodes.loc[j+1, 'Object']:
                                prev_node_value = node_value
                                prev_node_change_time = last_change_time_in_minutes
                                flag = 1
                        else:
                            node_num += 1
                            node_place_in_house = nodes.loc[j, 'place_in_house']
                            node_type = nodes.loc[j, 'Type']
                            feature.append([prev_node_value, node_place_in_house, node_type, prev_node_change_time])
                            if nodes.loc[j, 'Object'] != nodes.loc[j+1, 'Object']:
                                flag = 0

                    feature.append([house.loc[i, 'time_of_the_day'], -1, -1, -1])
                    g.ndata['attr'] = torch.tensor(feature)

                # Give Label
                    try:
                        mappedActivity = config['merging_activties'][house.iloc[i, 2]]
                        labels.append(getIDFromClassName(mappedActivity, config))
                    except:
                        activity = house.iloc[i, 2]
                        labels.append(getIDFromClassName(activity, config))

                    graphs.append(g)

                graph_labels = {"glabel": torch.tensor(labels)}

                save_graphs(graph_path, graphs, graph_labels)

            else:
                graphs, labels = load_graphs(graph_path)
                labels = list(labels['glabel'].numpy())
                if config['ob_data_Decompressed']:
                    DecompressedGraphs, DecompressedLabels = load_graphs(decompressedGraphPath)
                    DecompressedLabels = list(DecompressedLabels['glabel'].numpy())

            print(len(graphs))

            total_num_iteration_for_LOOCV = 0
            total_acc_for_LOOCV = []
            total_f1_for_LOOCV = []
            total_per_class_accuracy = []
            total_confusion_matrix = []
            score = 0
            accuracy = 0

            df = None
            # read csv Files
            house_name, all_test_loss, all_test_acc, all_test_f1_score, all_test_per_class_accuracy, all_test_confusion_matrix = [], [], [], [], [], []
            house_name_list = ['ordonezB', 'houseB', 'houseC', 'houseA', 'ordonezA']

            decompressed_csv = pd.read_csv(decompressed_csv_path)

            compressed_csv = pd.read_csv(ob_csv_file_path)

            uniqueIndex = getUniqueStartIndex(compressed_csv)


            # Required in case of ob Decompressed, when you want test index from
            # Decompressed csv rather than from OB CSV
            uniqueIndex_decompressed = getUniqueStartIndex(decompressed_csv)

            # Mapped Activity as per the config/generalizing the activities not present in all csvs'

            loo = LeaveOneOut()

            for train_index, test_index in loo.split(uniqueIndex):
                model = GIN(
                    args.num_layers, args.num_mlp_layers,
                    args.input_features, args.hidden_dim, args.nb_classes,
                    args.final_dropout, args.learn_eps,
                    args.graph_pooling_type, args.neighbor_pooling_type, args.save_embeddings).to(args.device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)



                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                # initialize the early_stopping object
                early_stopping = EarlyStopping(patience=15, verbose=True)

                print('----------------------------------------------------------------------------------------------')
                print('\n\n split: ', total_num_iteration_for_LOOCV)
                total_num_iteration_for_LOOCV += 1

                # Get start and end of test dataset
                start, end = getStartAndEndIndex(compressed_csv, uniqueIndex[test_index])
                # make dataframe for train, skip everything b/w test start and test end. rest everything is train.


                train_graphs = graphs[:start] + graphs[end:]
                train_labels = labels[:start] + labels[end:]

                # Divide train, test and val dataframe
                val_graphs = train_graphs[:int(len(train_graphs) * args.split_ratio)]
                val_labels = train_labels[:int(len(train_labels) * args.split_ratio)]

                train_graphs = train_graphs[int(len(train_graphs) * args.split_ratio):]
                train_labels = train_labels[int(len(train_labels) * args.split_ratio):]


                # Only Test index will be picked from decompressed because
                # Only while evaluating we are decompressing
                if config['ob_data_Decompressed']:
                    start, end = getStartAndEndIndex(decompressed_csv, uniqueIndex_decompressed[test_index])
                    test_graphs = DecompressedGraphs[start:end]
                    test_labels = DecompressedLabels[start:end]
                else:
                    test_graphs = graphs[start:end]
                    test_labels = labels[start:end]

                # Means this the last split and test has 1 element in it. skip it and continue, because this causes
                # the code to break. Kind of easy fix.
                if start == end:
                    continue

                trainDataset = GraphHouseDataset(train_graphs, train_labels)
                valDataset = GraphHouseDataset(val_graphs, val_labels)
                testDataset = GraphHouseDataset(test_graphs, test_labels)

                trainloader = GraphDataLoader(
                    trainDataset, batch_size=args.batch_size, device=args.device,
                    collate_fn=collate, seed=args.seed, shuffle=shuffle,
                    split_name='fold10', fold_idx=args.fold_idx, save_embeddings= args.save_embeddings).train_valid_loader()

                validloader = GraphDataLoader(
                    valDataset, batch_size=args.batch_size, device=args.device,
                    collate_fn=collate, seed=args.seed, shuffle=shuffle,
                    split_name='fold10', fold_idx=args.fold_idx, save_embeddings= args.save_embeddings).train_valid_loader()


                testloader = GraphDataLoader(
                    testDataset, batch_size=args.batch_size, device=args.device,
                    collate_fn=collate, seed=args.seed, shuffle=shuffle,
                    split_name='fold10', fold_idx=args.fold_idx, save_embeddings=args.save_embeddings).train_valid_loader()


                criterion = nn.CrossEntropyLoss()  # default reduce is true

                # Training
                training(model, trainloader, validloader, optimizer, criterion, scheduler, early_stopping)

                # Load Best Model from early stopping
                path = './checkpoint.pth'
                if os.path.isfile(path):
                    print("=> loading checkpoint '{}'".format(path))
                    checkpoint = torch.load(path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint)
                    # optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}'"
                          .format(path))
                else:
                    print("=> no checkpoint found at '{}'".format(path))

                if len(testloader) != 0:
                    test_loss, test_acc, test_f1_score, test_per_class_accuracy, test_confusion_matrix = eval_net(
                    args, model, testloader, criterion, text='test')

                    total_acc_for_LOOCV.append(test_acc)
                    total_f1_for_LOOCV.append(test_f1_score)
                    total_per_class_accuracy.append(test_per_class_accuracy)
                    total_confusion_matrix.append(test_confusion_matrix)

                    print('test set - average loss: {:.4f}, accuracy: {:.0f}%  test_f1_score: {:.4f} '
                          .format(test_loss, 100. * test_acc, test_f1_score))

            house_results_dictionary = {}

            print(file_name + '\n \n', 'test_acc:\t', np.mean(total_acc_for_LOOCV), '\t test f1 score',
                  np.mean(total_f1_for_LOOCV),
                  '\t test_per_class_accuracy: \n', dict(pd.DataFrame(total_per_class_accuracy).mean()))

            house_results_dictionary['accuracy'] = np.mean(total_acc_for_LOOCV)

            house_results_dictionary['f1_score'] = np.mean(total_f1_for_LOOCV)

            house_results_dictionary['test_per_class_accuracy'] = dict(pd.DataFrame(total_per_class_accuracy).mean())

            house_results_dictionary['confusion_matrix'] = total_confusion_matrix

            house_results_dictionary['house_name'] = file_name

            results_list.append(house_results_dictionary)

            if not os.path.exists(os.path.join('../../../logs', 'singleHouseGraphClassification')):
                os.mkdir(os.path.join('../../../logs', 'singleHouseGraphClassification'))

            print('\n\n\n\n\n\n Finished house', file_name, '\n\n\n\n')

        if config['ob_data_compressed']:
            print('saved')
            np.save(os.path.join('../../../logs/singleHouseGraphClassification' 'ob_compressed.npy'), results_list)
        elif config['ob_data_Decompressed']:
            print('saved')
            np.save(os.path.join('../../../logs/singleHouseGraphClassification' 'ob_decompressed.npy'), results_list)
        elif config['raw_data']:
            print('saved')
            np.save(os.path.join('../../../logs/singleHouseGraphClassification' 'raw.npy'), results_list)

def training(model, trainloader, validloader, optimizer, criterion, scheduler, early_stopping):
    for epoch in range(args.epochs):
        train(args, model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        # early_stopping needs the F1 score to check if it has increased,
        # and if it has, it will make a checkpoint of the current model

        if epoch % 10 == 9:
            print('epoch: ', epoch)
            train_loss, train_acc, train_f1_score, train_per_class_accuracy,_ = eval_net(
                args, model, trainloader, criterion)

            print('train set - average loss: {:.4f}, accuracy: {:.0f}%  train_f1_score: {:.4f} '
                  .format(train_loss, 100. * train_acc, train_f1_score))

            # print('train per_class accuracy', test_per_class_accuracy)

            valid_loss, valid_acc, val_f1_score, val_per_class_accuracy, _ = eval_net(
                args, model, validloader, criterion, text='val')

            print('valid set - average loss: {:.4f}, accuracy: {:.0f}% val_f1_score {:.4f}:  '
                  .format(valid_loss, 100. * valid_acc, val_f1_score))

            # print('val per_class accuracy', val_per_class_accuracy)

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_f1_score, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)

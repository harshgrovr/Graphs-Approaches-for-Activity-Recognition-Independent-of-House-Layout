import sys
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import os


from dataloader import GraphDataLoader, collate
from parser import Parser
from gin import GIN
import pandas as pd
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
         {'name': 'Idle', 'id': 7},
         {'name': 'getDrink', 'id': 8},
         {'name': 'prepareBreakfast', 'id': 9},
         {'name': 'getSnack', 'id': 10},
         {'name': 'idle', 'id': 11},
         {'name': 'storeGroceries', 'id': 12},
         {'name': 'washClothes', 'id': 13},
         {'name': 'grooming', 'id': 14},
         {'name': 'prepareDinner', 'id': 15},
         {'name': ' grooming', 'id': 16},
         {'name': 'relaxing', 'id': 17},
         {'name': 'useToilet', 'id': 18}],

    "merging_activties" : {
        "loadDishwasher": "washDishes",
        "unloadDishwasher": "washDishes",
        "loadWashingmachine": "washClothes",
        "unloadWashingmachine": "washClothes",
        "receiveGuest": "relaxing",
        "eatDinner": "eating",
        "eatBreakfast": "eating",
        "getDressed": " grooming",
        "shave": "grooming",
        "takeMedication": "Idle",
        "leave_Home": "leaveHouse",
        "Sleeping": "goToBed",
        "Bed_to_Toilet": "useToilet",
        "Enter_Home": "Idle",
        "Respirate": "relaxing",
        "Work": "Idle",
        "Housekeeping": "Idle",
        "watchTV": "relaxing"
    }
}
def getClassnameFromID(train_label):

    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["id"] == int(train_label)]
    return train_label[0]['name']

def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    # bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for (graphs, labels) in trainloader:
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr').to(args.device)
        graphs = graphs.to(args.device)
        outputs = net(graphs, feat)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        # bar.set_description('epoch-{}'.format(epoch))
    # bar.close()
    # the final batch will be aligned
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
    nb_classes = 19
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    for data in dataloader:
        graphs, labels = data
        feat = graphs.ndata.pop('attr').to(args.device)
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        total += len(labels)
        outputs = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

        all_labels.extend(labels.cpu())
        all_predicted.extend(predicted.cpu())

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    np.save('./' + text + '_confusion_matrix.npy', confusion_matrix)

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

    return loss, acc, f1, per_class_acc_dict

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


def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()
    is_cuda = False

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")


    model = GIN(
        args.num_layers, args.num_mlp_layers,
        4, args.hidden_dim, 19,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    file_names = ['houseA', 'houseB', 'houseC', 'ordonezA']

    graph_path  = os.path.join('../../../data/all_houses/all_houses.bin')

    graphs = []
    labels = []

    if not os.path.exists(graph_path):
     for file_name in file_names:
        print('\n\n\n\n')
        print('*******************************************************************')
        print('\t\t\t\t\t' + file_name + '\t\t\t\t\t\t\t')
        print('*******************************************************************')
        print('\n\n\n\n')
        house = pd.read_csv('../../../data/' + file_name + '/' + file_name + '.csv')
        nodes = pd.read_csv('../../../data/' + file_name + '/nodes.csv')
        edges = pd.read_csv('../../../data/' + file_name + '/bidrectional_edges.csv')
        lastChangeTimeInMinutes = pd.read_csv('../../../data/' + file_name + '/' + 'houseB' + '-sensorChangeTime.csv')

        u = edges['Src']
        v = edges['Dst']

        # Create Graph per row of the House CSV



        # Combine Feature like this: Place_in_House,Type, Value, Last_change_Time_in_Second for each node
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

                if flag == 0 :
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

    train_idx, valid_idx = _split_rand(labels)

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_graphs = [graphs[i] for i in valid_idx]
    val_labels = [labels[i] for i in valid_idx]

    trainDataset = GraphHouseDataset(train_graphs, train_labels)
    valDataset = GraphHouseDataset(val_graphs, val_labels)

    trainloader = GraphDataLoader(
        trainDataset, batch_size=args.batch_size, device=args.device,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()

    validloader = GraphDataLoader(
        valDataset, batch_size=args.batch_size, device=args.device,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()


    # or split_name='rand', split_ratio=0.7

    if os.path.exists('./saved_model/saved_model'):
        state = torch.load('./saved_model/saved_model')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    # tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    # vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    # lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch in range(args.epochs):
        train(args, model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        if epoch % 10 == 0:
            print('epoch: ', epoch)
            train_loss, train_acc, train_f1_score, train_per_class_accuracy = eval_net(
                args, model, trainloader, criterion)

            print('train set - average loss: {:.4f}, accuracy: {:.0f}%  train_f1_score: {:.4f} '
                    .format(train_loss, 100. * train_acc, train_f1_score))

            # print('train per_class accuracy', train_per_class_accuracy)

            valid_loss, valid_acc, val_f1_score, val_per_class_accuracy = eval_net(
                args, model, validloader, criterion, text='test')

            print('valid set - average loss: {:.4f}, accuracy: {:.0f}% val_f1_score {:.4f}:  '
                    .format(valid_loss, 100. * valid_acc, val_f1_score))

            # print('val per_class accuracy', val_per_class_accuracy)


            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(checkpoint, './saved_model/saved_model')

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                f.write("%f %f %f %f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))
                f.write("\n")

    #     lrbar.set_description(
    #         "Learning eps with learn_eps={}: {}".format(
    #             args.learn_eps, [layer.eps.data.item() for layer in model.ginlayers]))
    #
    # tbar.close()
    # vbar.close()
    # lrbar.close()


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)

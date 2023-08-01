import argparse
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from visdom import Visdom
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import models
import dataset
import pickle


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', add_help=False)
    parser.add_argument('--training_mode', default='wa', type=str)
    parser.add_argument('--num_bases', default=5, type=int)
    parser.add_argument('--num_feature', default=80, type=int)
    parser.add_argument('--increment', default=3, type=int)
    parser.add_argument('--num_tasks', default=6, type=int)
    parser.add_argument('--backbone', default="lstm", type=str)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--input_size', default=8, type=int)
    parser.add_argument('--task_id', default=0, type=int)
    parser.add_argument('--herding_method', default="loe", type=str)
    parser.add_argument('--base_mem_size', default=5, type=int)
    parser.add_argument('--inc_mem_size', default=5, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--lambda_kd', default=0, type=float)
    parser.add_argument('--patience', default=100, type=int)
    return parser


parser = get_args_parser()
args = parser.parse_args()


def build_trainset(encoding_data, args):
    '''
        divide training set into different tasks
    '''
    scenario_train = []
    for task_id in range(args.num_tasks):
        task = []
        if task_id == 0:
            for sample in encoding_data:
                if sample[1] <= args.num_bases-1:
                    task.append(sample)
            scenario_train.append(task)
        else:
            for sample in encoding_data:
                if args.num_bases + (task_id-1)*args.increment-1 < sample[1] <= args.num_bases + task_id*args.increment-1:
                    task.append(sample)
            scenario_train.append(task)
    return scenario_train

def build_testset(encoding_data, args):
    '''
        divide testing set into different tasks
    '''
    scenario_test = []
    for task_id in range(args.num_tasks):
        task = []
        for sample in encoding_data:
            if sample[1] <= args.num_bases + task_id*args.increment - 1:
                task.append(sample)
        scenario_test.append(task)
    return scenario_test

def save_as_txt(test_results, test_results2, train_results, save_path):
    file = open(save_path,'w')
    file.write('Test Results\n')
    for result in test_results:
        file.write(str(result) + '\n')

    file.write('Test after WA\n')
    for result in test_results2:
        file.write(str(result) + '\n')

    file.write('Train Results\n')
    for result in train_results:
        file.write(str(result) + '\n')
    file.close()

def eval_model(model, val_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            model.eval()
            X_batch = batch_data['sequences'].to(device)
            y_batch = batch_data['label'].to(device).long()
            X_seq_len = batch_data['seq_len']
            y_pred, _ = model(X_batch, X_seq_len)

            class_predictions = F.softmax(y_pred, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (class_predictions == y_batch).sum().item()
    acc = correct / total

    return acc


def model_training(encoding_data_train, encoding_data_test, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenario_train = build_trainset(encoding_data_train, args)
    scenario_test = build_testset(encoding_data_test, args)


    model = models.CilModel(args)
    model = model.cuda()

    teacher_model = None
    criterion = nn.CrossEntropyLoss()
    kd_criterion = models.SoftTarget(T=2)
    args.known_classes = 0
    rehearsal = models.Rehearsal(args)

    task_acc = []
    task_acc2 = []
    task_acc_train = []

    print('Start model training')
    for task_id, train_sequence in enumerate(scenario_train):

        args.task_id = task_id
        train_sequence_raw = train_sequence.copy()
        train_sequence += rehearsal.memory
        train_set = dataset.AppsDataset(train_sequence)
        val_set = dataset.AppsDataset(scenario_test[task_id])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_set, shuffle=True, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

        # Training Loop
        if args.training_mode == 'joint':
            model = models.CilModel(args)
            model = model.cuda()
            model.prev_model_adaption(args.num_bases+task_id*args.increment, args)

        else:
            if task_id == 0:
                model.prev_model_adaption(args.num_bases, args)
            else:
                model.prev_model_adaption(args.increment, args)

        args.lambda_kd = args.known_classes/(args.known_classes+args.increment)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # viz = Visdom()
        # viz.line([0.], [0.], win='Train_loss', opts=dict(title='Train_loss'))
        # viz.line([0.], [0.], win='Train_acc', opts=dict(title='Train_acc'))
        # viz.line([0.], [0.], win='Val_loss', opts=dict(title='Val_loss'))
        # viz.line([0.], [0.], win='Val_acc', opts=dict(title='Val_acc'))

        best_acc = 0
        best_acc_train = 0
        patience = args.patience
        patience_counter = 0

        for epoch in tqdm(range(args.num_epochs)):
            loss_train_total = 0
            loss_val_total = 0
            correct_train, total_train = 0, 0
            # Training Loop
            for idx, batch_data in enumerate(train_loader):
                model.train()
                X_batch = batch_data['sequences'].to(device)
                y_batch = batch_data['label'].to(device).long()
                X_seq_len = batch_data['seq_len']

                y_pred,_ = model(X_batch, X_seq_len)
                loss_ce = criterion(y_pred, y_batch)
                if teacher_model is not None:
                    t_pred, _ = teacher_model(X_batch, X_seq_len)
                    loss_kd = kd_criterion(y_pred[:, :args.known_classes], t_pred)
                else:
                    loss_kd = torch.tensor(0.).to(device)
                loss = args.lambda_kd*loss_kd + (1-args.lambda_kd)*loss_ce

                class_predictions = F.softmax(y_pred, dim=1).argmax(dim=1)
                total_train += y_batch.size(0)
                correct_train += (class_predictions == y_batch).sum().item()
                loss_train_total += loss.cpu().detach().item() * args.batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_train_total = loss_train_total / len(train_set)
            acc_train = correct_train / total_train

            # viz.line([loss_train_total], [epoch], win='Train_loss', update='append')
            # viz.line([acc_train], [epoch], win='Train_acc', update='append')

            # Validation Loop
            correct, total = 0, 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()
                    X_batch = batch_data['sequences'].to(device)
                    y_batch = batch_data['label'].to(device).long()
                    X_seq_len = batch_data['seq_len']
                    y_pred, _ = model(X_batch, X_seq_len)

                    loss = criterion(y_pred, y_batch)
                    loss_val_total += loss.cpu().detach().item() * args.batch_size
                    class_predictions = F.softmax(y_pred, dim=1).argmax(dim=1)

                    total += y_batch.size(0)
                    correct += (class_predictions == y_batch).sum().item()

            loss_val_total = loss_val_total / len(val_set)
            acc = correct / total

            # viz.line([loss_val_total], [epoch], win='Val_loss', update='append')
            # viz.line([acc], [epoch], win='Val_acc', update='append')

            # Best acc update and early stop check
            if acc > best_acc:
                patience_counter = 0
                best_acc = acc
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f'Early stopping on epoch {epoch} with accuracy: {best_acc:2.2%} and train accuracy {best_acc_train:2.2%}')
                    break

            if acc_train > best_acc_train:
                best_acc_train = acc_train

        # Update args
        if task_id == 0:
            args.known_classes += args.num_bases
        else:
            args.known_classes += args.increment

        if args.training_mode == 'wa':
            # Weight Alignment
            model.after_model_adaption(args.increment, args) # Weight Alignment
            acc2 = eval_model(model, val_loader, device)
            task_acc2.append(acc2)

            # Record teacher model
            teacher_model = model.copy().freeze()

            # Feature extraction
            features = []
            train_set_raw = dataset.AppsDataset(train_sequence_raw)
            unshuffle_train_loader = DataLoader(train_set_raw, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
            for idx, batch_data in enumerate(unshuffle_train_loader):
                X_batch = batch_data['sequences'].to(device)
                X_seq_len = batch_data['seq_len']
                features.append(model.extract_vector(X_batch, X_seq_len).detach().cpu().numpy())
            features = np.concatenate(features, axis=0)

            entropy = []
            unshuffle_train_loader = DataLoader(train_set_raw, shuffle=False, batch_size=1, collate_fn=dataset.collate_fn)
            for idx, batch_data in enumerate(unshuffle_train_loader):
                X_batch = batch_data['sequences'].to(device)
                X_seq_len = batch_data['seq_len']
                y_batch = batch_data['label'].to(device).long()
                y_pred, _ = model(X_batch, X_seq_len)
                entropy.append(criterion(y_pred, y_batch).detach().cpu().numpy())

            # Rehearsal
            rehearsal.add_sample(train_sequence_raw, args.task_id, features, entropy)

        if args.training_mode == 'joint':
            # Record the whole training set
            rehearsal.add_all(train_sequence_raw)
            task_acc2 = 0

        task_acc.append(best_acc)
        task_acc_train.append(best_acc_train)
        print('task% d training complete'%task_id)
    return task_acc, task_acc2, task_acc_train


if __name__ == "__main__":
    save_path1 = 'encoded__train_data.pkl'  # Encoded data pkl file
    save_path2 = 'encoded__test_data.pkl'  # Encoded data pkl file
    with open(save_path1, 'rb') as ff:
        train_data = pickle.load(ff)
        ff.close()
    with open(save_path2, 'rb') as ff:
        test_data = pickle.load(ff)
        ff.close()

    # random herding
    args.training_mode = 'wa'
    args.herding_method = 'random'
    rand_acc_list = []
    rand_acc2_list = []
    rand_acc_train_list = []
    for i in range(1):
        rand_acc, ran_acc2, rand_acc_train = model_training(train_data, test_data, args)
        print('Val acc: ', rand_acc)
        print('Val acc with WA', ran_acc2)
        print('Train acc:', rand_acc_train)
        rand_acc_list.append(rand_acc)
        rand_acc2_list.append(ran_acc2)
        rand_acc_train_list.append(rand_acc_train)
    save_as_txt(rand_acc_list, rand_acc2_list, rand_acc_train_list, './Results/rand_results.txt')

    # # nem herding
    # args.training_mode = 'wa'
    # args.herding_method = 'nem'
    # nem_acc_list = []
    # nem_acc_list2 = []
    # nem_acc_train_list = []
    # for i in range(5):
    #     nem_acc, nem_acc2, nem_acc_train = model_training(train_data, test_data, args)
    #     print('Val acc: ', nem_acc)
    #     print('Train acc:', nem_acc_train)
    #     nem_acc_list.append(nem_acc)
    #     nem_acc_list2.append(nem_acc2)
    #     nem_acc_train_list.append(nem_acc_train)
    # save_as_txt(nem_acc_list, nem_acc_list2, nem_acc_train_list, './Results/nem_results.txt')
    #
    # # loe herding
    # args.training_mode = 'wa'
    # args.herding_method = 'loe'
    # loe_acc_list = []
    # loe_acc_list2 = []
    # loe_acc_train_list = []
    # for i in range(5):
    #     loe_acc, loe_acc2, loe_acc_train = model_training(train_data, test_data, args)
    #     print('Val acc: ', loe_acc)
    #     print('Train acc:', loe_acc_train)
    #     loe_acc_list.append(loe_acc)
    #     loe_acc_list2.append(loe_acc2)
    #     loe_acc_train_list.append(loe_acc_train)
    # save_as_txt(loe_acc_list, loe_acc_list2, loe_acc_train_list,'./Results/loe_results.txt')
    #
    # # Ft training
    # args.training_mode = 'ft'
    # args.patience = 50
    # ft_acc_list = []
    # ft_acc_train_list = []
    # for i in range(5):
    #     ft_acc, _, ft_acc_train = model_training(train_data, test_data, args)
    #     print('Val acc for %d round:' % i, ft_acc)
    #     print('Train acc for %d round:' % i, ft_acc_train)
    #     ft_acc_list.append(ft_acc)
    #     ft_acc_train_list.append(ft_acc_train)
    # save_as_txt(ft_acc_list, ft_acc_list, ft_acc_train_list,'./Results/ft_results.txt')
    #
    # # Joint training
    # args.training_mode = 'joint'
    # args.patience = 50
    # joint_acc_list = []
    # joint_acc_train_list = []
    # for i in range(5):
    #     joint_acc, _, joint_train_acc = model_training(train_data, test_data, args)
    #     print('Val acc for %d round:' % i, joint_acc)
    #     print('Train acc for %d round:' % i, joint_train_acc)
    #     joint_acc_list.append(joint_acc)
    #     joint_acc_train_list.append(joint_train_acc)
    # save_as_txt(joint_acc_list, joint_acc_list, joint_acc_train_list,'./Results/joint_results.txt')
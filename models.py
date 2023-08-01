import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import numpy as np

def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad

class LSTMbackbone(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=80, num_layers=2,
                 dropout=0):
        """
        input_dim = number of features at each time step
                    (number of features given to each LSTM cell)
        hidden_dim = number of features produced by each LSTM cell (in each layer)
        num_layers = number of LSTM layers
        output_dim = number of classes of the floor texture
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False,)
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, X_seq_len):
        self.lstm.flatten_parameters()
        hidden_features, (_, _) = self.lstm(X)  # (h_0, c_0) default to zeros
        out_pad, out_len = pad_packed_sequence(hidden_features, batch_first=True)
        lstm_out_forward = out_pad[range(len(out_pad)), X_seq_len-1, :self.hidden_dim]

        # out = self.fc(lstm_out_forward)
        return lstm_out_forward

    def out_dim(self):
        return self.hidden_dim

class CilClassifier(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes).cuda()])

    def __getitem__(self, index):
        return self.heads[index]

    def __len__(self):
        return len(self.heads)

    def forward(self, x):
        logits = torch.cat([head(x) for head in self.heads], dim=1)
        return logits

    def adaption(self, nb_classes):
        self.heads.append(nn.Linear(self.embed_dim, nb_classes).cuda())

class CilModel(nn.Module):
    def __init__(self, args):
        super(CilModel, self).__init__()
        self.backbone = LSTMbackbone(hidden_dim=args.num_feature)
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x, x_len):
        return self.backbone(x, x_len)

    def forward(self, x, x_len):
        x = self.backbone(x, x_len)
        out = self.fc(x)
        return out, x

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self, names=["all"]):
        freeze_parameters(self, requires_grad=True)
        self.train()
        for name in names:
            if name == 'fc':
                freeze_parameters(self.fc)
                self.fc.eval()
            elif name == 'backbone':
                freeze_parameters(self.backbone)
                self.backbone.eval()
            elif name == 'all':
                freeze_parameters(self)
                self.eval()
            else:
                raise NotImplementedError(
                    f'Unknown module name to freeze {name}')
        return self

    def prev_model_adaption(self, nb_classes, args):
        if self.fc is None:
            self.fc = CilClassifier(args.num_feature, nb_classes).cuda()
        else:
            self.fc.adaption(nb_classes)

    def after_model_adaption(self, nb_classes, args):
        if args.task_id > 0:
            self.weight_align(nb_classes)

    @torch.no_grad()
    def weight_align(self, nb_new_classes):
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1)

        norm_old = norms[:-nb_new_classes]
        norm_new = norms[-nb_new_classes:]

        gamma = torch.mean(norm_old) / torch.mean(norm_new)
        print(f"old norm / new norm ={gamma}")
        self.fc[-1].weight.data = gamma * w[-nb_new_classes:]

class SoftTarget(nn.Module):

    def __init__(self, T=2):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss

def random_add(task_id, base_mem_size, inc_mem_size):
    select_id = np.array([])
    if task_id == 0:
        num_classes = 5
        num_samples = 90
        num_per_class = base_mem_size
    else:
        num_classes = 3
        num_samples = 10
        num_per_class = inc_mem_size
    for class_id in range(num_classes):
        sample_id = list(range(class_id * num_samples, (class_id + 1) * num_samples))
        select_id = \
            np.concatenate((select_id, np.random.choice(sample_id, num_per_class)), axis=0)
    return select_id

def nem_add(task_id, features: np.ndarray, base_mem_size, inc_mem_size):
    """Herd the samples whose features is the closest to their moving barycenter.

       Reference:
           * iCaRL: Incremental Classifier and Representation Learning
             Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
             CVPR 2017

       :param task_id: the id of current task
       :param features: Features of current training set with shape (nb_samples, nb_dim).
       :param base_mem_size: Number of samples to herd per base class.
       :param inc_mem_size: Number of samples to herd per incremental class.
       :return: The sampled data indices.
       """
    if len(features.shape) != 2:
        raise ValueError(f"Expected features to have 2 dimensions, not {len(features.shape)}d.")

    indexes = []

    if task_id == 0:
        num_classes = 5
        num_samples = 90
        num_per_class = base_mem_size
    else:
        num_classes = 3
        num_samples = 10
        num_per_class = inc_mem_size

    for class_id in range(num_classes):
        class_indexes = np.array(range(class_id * num_samples, (class_id + 1) * num_samples))
        class_features = features[class_indexes]

        D = class_features.T
        D = D / (np.linalg.norm(D, axis=0) + 1e-8)
        mu = np.mean(D, axis=1)
        herding_matrix = np.zeros((class_features.shape[0],))

        w_t = mu
        iter_herding, iter_herding_eff = 0, 0

        while not (
            np.sum(herding_matrix != 0) == min(num_per_class, class_features.shape[0])
        ) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if herding_matrix[ind_max] == 0:
                herding_matrix[ind_max] = 1 + iter_herding
                iter_herding += 1

            w_t = w_t + mu - D[:, ind_max]

        herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

        tmp_indexes = herding_matrix.argsort()[:num_per_class]
        indexes.append(class_indexes[tmp_indexes])

    indexes = np.concatenate(indexes)
    return indexes

def n2c_add(task_id, features: np.ndarray, base_mem_size, inc_mem_size):
    """Herd the samples whose features is the closest to their class mean.

    :param task_id: The id of current task
    :param features: Features of current training set with shape (nb_samples, nb_dim).
    :param base_mem_size: Number of samples to herd per base class.
    :param inc_mem_size: Number of samples to herd per incremental class.
    :return: The sampled data indices.
    """
    indexes = []
    if task_id == 0:
        num_classes = 5
        num_samples = 90
        num_per_class = base_mem_size
    else:
        num_classes = 3
        num_samples = 10
        num_per_class = inc_mem_size

    for class_id in range(num_classes):
        class_indexes = np.array(range(class_id * num_samples, (class_id + 1) * num_samples))
        class_features = features[class_indexes]
        class_mean = np.mean(class_features, axis=0, keepdims=True)

        dist_to_mean = np.linalg.norm(class_mean - class_features, axis=1)
        tmp_indexes = dist_to_mean.argsort()[:num_per_class]

        indexes.append(class_indexes[tmp_indexes])

    indexes = np.concatenate(indexes)
    return indexes

def loe_add(task_id, entropy, base_mem_size, inc_mem_size):
    """Herd the samples with lowest cross entropy.

    :param task_id: The id of current task
    :param entropy: entropy of each training sample from
                    current training set, with shape (nb_samples, nb_dim).
    :param base_mem_size: Number of samples to herd per base class.
    :param inc_mem_size: Number of samples to herd per incremental class.
    :return: The sampled data indices.
    """
    entropy = np.array(entropy)
    indexes = []
    if task_id == 0:
        num_classes = 5
        num_samples = 90
        num_per_class = base_mem_size
    else:
        num_classes = 3
        num_samples = 10
        num_per_class = inc_mem_size

    for class_id in range(num_classes):
        class_indexes = np.array(range(class_id * num_samples, (class_id + 1) * num_samples))
        class_entropy = entropy[class_indexes]
        tmp_indexes = class_entropy.argsort()[:num_per_class]
        indexes.append(class_indexes[tmp_indexes])

    indexes = np.concatenate(indexes)
    return indexes

class Rehearsal():
    def __init__(self, args):
        self.base_mem_size = args.base_mem_size
        self.inc_mem_size = args.inc_mem_size
        self.herding_method = args.herding_method
        self.memory = []

    def add_sample(self, train_seq, task_id, features, entropy):
        if self.herding_method == 'random':
            selected_id = random_add(task_id, self.base_mem_size, self.inc_mem_size)
        elif self.herding_method == 'nem':
            selected_id = nem_add(task_id, features, self.base_mem_size, self.inc_mem_size)
        elif self.herding_method == 'n2c':
            selected_id = nem_add(task_id, features, self.base_mem_size, self.inc_mem_size)
        elif self.herding_method == 'loe':
            selected_id = loe_add(task_id, entropy, self.base_mem_size, self.inc_mem_size)
        else:
            print('Not implemented yet')
        for idx in selected_id:
            self.memory.append(train_seq[int(idx)])

    def add_all(self, train_seq):
        self.memory += train_seq

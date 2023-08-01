from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch

class AppsDataset(Dataset):
    def __init__(self, data_sequences):
        self.data_sequences = data_sequences

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, index):
        isequences, ilabel = self.data_sequences[index]
        isequences = isequences.to_numpy()
        isequences = isequences[:, 1:9]
        return_dict = {'sequences': torch.from_numpy(isequences).float(), 'label': ilabel}
        return return_dict

class AppsDatasetHalf(Dataset):
    def __init__(self, data_sequences):
        self.data_sequences = data_sequences

    def __len__(self):
        return int(len(self.data_sequences)/2)

    def __getitem__(self, index):
        isequences, ilabel = self.data_sequences[index]
        isequences = isequences.to_numpy()
        isequences = isequences[:, 1:9]
        return_dict = {'sequences': torch.from_numpy(isequences).float(), 'label': ilabel}
        return return_dict


def collate_fn(data_label):
    data = [d['sequences'] for d in data_label]
    labels = [d['label'] for d in data_label]
    data_with_label = [(data[i], labels[i]) for i in range(len(data))]
    data_with_label.sort(key=lambda x: len(x[0]), reverse=True)
    data = [data_with_label[i][0] for i in range(len(data_with_label))]
    labels = [data_with_label[i][1] for i in range(len(data_with_label))]
    seq_len = [s.size(0) for s in data]  # 获取数据真实的长度
    data = pad_sequence(data, batch_first=True)
    # max_data = torch.max(torch.max(data, 0).values, 0)
    # data = data / max_data.values
    data = pack_padded_sequence(data, seq_len, batch_first=True)
    labels = torch.tensor(labels)
    seq_len = torch.tensor(seq_len)
    return {'sequences': data, 'label': labels, 'seq_len': seq_len}

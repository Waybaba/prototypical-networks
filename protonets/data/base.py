import torch

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class ODGSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, mode):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.mode = mode

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        # total_rot_num = 7
        self.n_real_classes = int(self.n_classes / 7)
        for _ in range(self.n_episodes):
            idxs = torch.randperm(self.n_real_classes)[:self.n_way]
            if self.mode == 'odg_train':
                yield (idxs * 7) + torch.randint(0, 5, size=(1,))
            elif self.mode == 'odg_test':
                yield (idxs * 7) + 5
            else:
                raise NameError(self.mode)
            
        # for i in range(self.n_episodes):
        #     yield torch.randperm(self.n_classes)[:self.n_way]
        # TODO return idxs with same angles
        # TODO test: return idxs with angle 75^0
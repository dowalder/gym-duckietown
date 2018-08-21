import torch.nn


class LabelsToCrossEntropy:

    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, result, label):
        return self.loss(result, label.long().squeeze())


class MySpecialMSELoss:

    def __init__(self):
        self.loss = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax()

    def __call__(self, result, label):
        out = self.softmax(result)
        dist = torch.zeros(out.shape, device=out.device)
        for idx, lbl in enumerate(label):
            lbl = int(lbl.item())
            if lbl == 0:
                dist[idx, 0] = 0.6
                dist[idx, 1] = 0.3
                dist[idx, 2] = 0.1
            elif lbl == out.shape[1] - 1:
                dist[idx, lbl - 1] = 0.1
                dist[idx, lbl - 1] = 0.3
                dist[idx, lbl] = 0.6
            else:
                dist[idx, lbl - 1] = 0.2
                dist[idx, lbl] = 0.6
                dist[idx, lbl + 1] = 0.2
        return self.loss(result, dist)

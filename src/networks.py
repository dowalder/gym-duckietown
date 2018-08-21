#!/usr/bin/env python3
import copy
import pathlib
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet

import src.resnet
import src.params
import src.options


class BaseModel(nn.Module):

    def __init__(self, params: Union[src.params.TrainParams, src.params.TestParams]):
        super(BaseModel, self).__init__()
        self.iteration = 0

        self.params = params

        self.train = False
        self.optim = None
        self.criterion = None

    def forward(self, x):
        raise NotImplemented()

    def training_step(self, samples, labels):
        self._check_training()

        self.optim.zero_grad()
        out = self.forward(samples)
        loss = self.criterion(out, labels)
        loss.backward()
        self.optim.step()
        self.iteration += 1
        return loss.item()

    def testing_step(self, samples, labels):
        self._check_training()

        with torch.no_grad():
            out = self.forward(samples)
            loss = self.criterion(out, labels)
        return loss.item()

    def init_training(self):
        self.optim = src.options.choose_optimizer(
            self.params.optimizer, self.parameters(), **self.params.optimizer_params)

        self.criterion = src.options.choose_criterion(self.params.criterion, **self.params.criterion_params)
        self.apply(weights_init)
        (self.params.model_path / "conf.yaml").write_text(self.params.conf_file_text())
        self.train = True

    def save(self, verbose=False) -> None:
        path = self.params.model_path / "{0:07d}_net.pth".format(self.iteration)
        torch.save(self.state_dict(), path.as_posix())
        if verbose:
            self.say("Saved model at {}".format(path))

    def _load_from_path(self, path: pathlib.Path):
        self.load_state_dict(torch.load(path.as_posix(), map_location=str(self.params.device)))

    def load(self, iteration=-1):
        if iteration == -1:
            paths = [path for path in self.params.model_path.glob("*.pth")]
            if not paths:
                raise FileNotFoundError(
                    "Could not load model weights. No .pth file found in {}".format(self.params.model_path))
            paths.sort()
            path = paths[-1]
            try:
                self.iteration = int(path.name.split("_")[0])
            except Exception as e:
                print("Could not load the last iteration number. Starting from 0: {}".format(e))

        else:
            path = self.params.model_path / "{0:07d}_net.pth}".format(iteration)
            self.iteration = iteration

        assert path.is_file(), "Could not find the model weights for iteration {}: {}".format(iteration, path)
        self._load_from_path(path)

    def say(self, msg: str) -> None:
        with (self.params.model_path / "out.txt").open("a") as fid:
            fid.write(msg + "\n")
        print(msg)

    def _check_training(self):
        if not self.train:
            raise RuntimeError("Training has not been initialized. Call init_training() first.")


def conv_stage(k_size: Tuple[int, ...],
               in_channels: int,
               out_channels: int,
               channels: Optional[Tuple[int, ...]]=None,
               stride: Union[Tuple[int, ...], int]=1,
               padding_size: Optional[Union[Tuple[int, ...], int]]=None,
               padding_type: str="zero",
               pooling: Optional[Tuple[bool, ...]]=None,
               norm_layer: Optional[str]="batch",
               bias=True,
               relu=True) -> nn.Module:

    assert isinstance(k_size, tuple)
    num_stages = len(k_size)

    if isinstance(stride, int):
        s = num_stages * [stride]
    else:
        assert len(stride) == num_stages
        s = stride

    if channels is not None:
        assert num_stages - 1 == len(channels)
        in_chs = (in_channels,) + channels
        out_chs = channels + (out_channels,)
    else:
        in_chs = num_stages * [out_channels]
        out_chs = in_chs
        in_chs[0] = in_channels

    if padding_size is None:
        pad = tuple(num_stages * [0])
    elif isinstance(padding_size, int):
        pad = tuple(num_stages * [padding_size])
    else:
        assert len(padding_size) == num_stages
        pad = padding_size

    if padding_type == "zero":
        pad_func = torch.nn.ZeroPad2d
    else:
        raise ValueError("Unknown padding layer: {}".format(padding_type))

    if pooling is None:
        pool = num_stages * [False]
    else:
        assert len(pooling) == num_stages
        pool = pooling

    if norm_layer is None:
        norm = None
    elif norm_layer == "batch":
        norm = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm = nn.InstanceNorm2d
    else:
        raise ValueError("Unknown normalization layer: {}".format(norm_layer))

    model = []
    for i in range(num_stages):
        if pad[i]:
            model += [pad_func(pad[i])]
        model += [nn.Conv2d(in_chs[i], out_chs[i], k_size[i], s[i], bias=bias)]
        if pool[i]:
            model += [nn.MaxPool2d(2)]
        if norm is not None:
            model += [norm(out_chs[i])]
        if relu:
            model += [nn.ReLU(True)]

    return nn.Sequential(*model)


def _num_flat_features(x):
    """
    Compute the number of features in a tensor.
    :param x:
    :return:
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)


class BasicLaneFollower(BaseModel):

    def __init__(self, params: Union[src.params.TrainParams, src.params.TestParams]):
        super(BasicLaneFollower, self).__init__(params=params)
        color = params.get("color", no_raise=True)
        in_channels = 1 if color == "gray" else 3
        self.conv = conv_stage(k_size=(7, 5, 5, 5),
                               in_channels=in_channels,
                               out_channels=64,
                               channels=(16, 32, 64),
                               stride=2,
                               padding_size=(3, 2, 2, 2),
                               relu=True,
                               norm_layer=None)
        self.fc1 = nn.Linear(10 * 5 * 64, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActionEstimator(BaseModel):

    def __init__(self, params: Union[src.params.TrainParams, src.params.TestParams]):
        super(ActionEstimator, self).__init__(params=params)
        color = params.get("color", no_raise=True)
        in_channels = 1 if color == "gray" else 3

        num_actions = params.get("num_actions", no_raise=False)
        assert isinstance(num_actions, int), num_actions.__class__

        self.conv = conv_stage(k_size=(7, 5, 5, 5),
                               in_channels=in_channels,
                               out_channels=64,
                               channels=(16, 32, 64),
                               stride=2,
                               padding_size=(3, 2, 2, 2),
                               relu=True,
                               norm_layer=None)

        self.fc1 = nn.Linear(10 * 5 * 64, 1024)
        self.fc2 = nn.Linear(1024, num_actions)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class IntentionTranslator(BaseModel):

    def __init__(self, params: Union[src.params.TrainParams, src.params.TestParams]):
        super(IntentionTranslator, self).__init__(params=params)
        color = params.get("color", no_raise=True)
        in_channels = 1 if color == "gray" else 3

        num_actions = params.get("num_actions", no_raise=False)
        assert isinstance(num_actions, int), num_actions.__class__

        self.conv = conv_stage(k_size=(7, 5, 5, 5),
                               in_channels=in_channels,
                               out_channels=64,
                               channels=(16, 32, 64),
                               stride=2,
                               padding_size=(3, 2, 2, 2),
                               relu=True,
                               norm_layer=None)

        action_estimator_weights = params.get("action_estimator_weights", no_raise=False)
        self.action_estimator = ActionEstimator(copy.deepcopy(params))
        self.action_estimator.params.model_path = action_estimator_weights
        self.action_estimator.load()

        self.fc1 = nn.Linear(10 * 5 * 64, 1024)

        hidden_size = params.get("hidden_size", default=128)
        num_layers = params.get("num_layers", default=1)
        self.lstm = nn.LSTM(input_size=1024 + num_actions, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x_in):
        with torch.no_grad():
            actions = self.action_estimator(x_in)

        x = self.conv(x_in)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.cat(0, (x, actions))
        x = self.lstm(x)
        return x


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(15 * 10 * 64, 1024)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class InitialNet(nn.Module):
    """
    The exact (as possible) copy of the caffe net in https://github.com/syangav/duckietown_imitation_learning
    """

    def __init__(self):
        super(InitialNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=2)
        # self.dropout = nn.Dropout2d(p=0.3)

        self.fc1 = nn.Linear(10 * 5 * 64, 1024)
        self.fc2 = nn.Linear(1024, 1)  # TODO: change back

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = self.dropout(x)
        x = x.view(-1, _num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.tanh(x)
        return x


class InitialNetSharingWeights(InitialNet):

    def __init__(self):
        super(InitialNetSharingWeights, self).__init__()

    def conv_only(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

    def first_fc(self, img):
        x = self.conv_only(img)
        x = x.view(-1, _num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x


class ResnetController(nn.Module):

    def __init__(self):
        super(ResnetController, self).__init__()

        self.resnet = src.resnet.resnet8()

        self.dropout = nn.Dropout2d(0.3)

        self.fc1 = nn.Linear(1000, 250)
        self.fc2 = nn.Linear(250, 125)
        self.fc3 = nn.Linear(125, 2)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NImagesNet(InitialNet):
    """
    This extends the InitialNet so that it accepts larger images. Images for this network can have n times more pixels
    in the height, but need to keep the original width.
    """

    def __init__(self, n):
        super(NImagesNet, self).__init__()
        self.fc1 = nn.Linear(n * 10 * 5 * 64, 1024)


class DoubleLabelNet(NImagesNet):
    """
    Is able to give out the current and the next label as an output of the network.
    """

    def __init__(self, n):
        super(DoubleLabelNet, self).__init__(n=n)
        self.fc2 = nn.Linear(1024, 2)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class WeightsSharingRNN(nn.Module):

    def __init__(self, cnn_weights_path: Optional[pathlib.Path]=None, num_lstms=1, cnn_no_grad=False, device="cpu"):
        super(WeightsSharingRNN, self).__init__()

        self.cnn = InitialNetSharingWeights()
        if cnn_weights_path is not None:
            self.cnn.load_state_dict(torch.load(cnn_weights_path.as_posix()))
        self.cnn_no_grad = cnn_no_grad

        self.num_lstms = num_lstms
        self.rnn = nn.LSTM(input_size=1026, hidden_size=128, num_layers=self.num_lstms, dropout=0.0)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = torch.device(device)

        self.init_hidden()

    def to(self, device):
        super(WeightsSharingRNN, self).to(device)
        self.device = torch.device(device)
        self.hidden = (self.hidden[0].to(self.device), self.hidden[1].to(self.device))

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def forward(self, img, action):
        if self.cnn_no_grad:
            with torch.no_grad():
                x = self.cnn.first_fc(img)
        else:
            x = self.cnn.first_fc(img)
        x = torch.cat((x, action), 1)
        x = x.unsqueeze(1)
        x, self.hidden = self.rnn(x, self.hidden)
        x = self.fc_final(x)
        return x


class BasicConvRNN(nn.Module):

    def __init__(self, device="cpu"):
        super(BasicConvRNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 11 * 16, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)

        self.num_lstms = 1
        self.rnn = nn.LSTM(input_size=1002, hidden_size=128, num_layers=self.num_lstms)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = device
        self.init_hidden()

    def to(self, device):
        super(BasicConvRNN, self).to(device)
        self.device = device

    def cnn_pass(self, img):
        out = F.relu(self.conv1(img))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def rnn_pass(self, x, action):
        if len(action.shape) <= 1:
            action = action.unsqueeze(0)

        out = torch.cat((x, action), 1)
        out = out.unsqueeze(1)
        out, _ = self.rnn(out, self.hidden)
        out = self.fc_final(out)
        return out

    def forward(self, img, action):
        out = self.cnn_pass(img)
        out = self.rnn_pass(out, action)
        return out


class ResnetRNN(nn.Module):

    def __init__(self, pretrained=False, device="cpu"):
        super(ResnetRNN, self).__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=pretrained)

        self.num_lstms = 1
        self.rnn = nn.LSTM(input_size=1002, hidden_size=128, num_layers=self.num_lstms)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = device
        self.init_hidden()

    def to(self, device):
        super(ResnetRNN, self).to(device)
        self.device = device

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def forward(self, img, action):
        x = self.resnet(img)

        if len(action.shape) <= 1:
            action = action.unsqueeze(0)

        x = torch.cat((x, action), 1)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x, self.hidden)
        x = self.fc_final(x)
        return x


class ResnetRNNsmall(nn.Module):

    def __init__(self, pretrained=False, device="cpu"):
        super(ResnetRNNsmall, self).__init__()
        self.num_lstms = 1

        self.resnet = src.resnet.resnet8()
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = nn.LSTM(input_size=1002, hidden_size=128, num_layers=self.num_lstms)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = device
        self.init_hidden()

    def to(self, device):
        super(ResnetRNNsmall, self).to(device)
        self.device = device

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def forward(self, img, action):
        x = self.resnet(img)
        x = self.dropout(x)

        if len(action.shape) <= 1:
            action = action.unsqueeze(0)

        x = torch.cat((x, action), 1)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x, self.hidden)
        x = self.fc_final(x)
        return x


class SequenceCnn(nn.Module):

    def __init__(self, device="cpu", num_imgs=5):
        super(SequenceCnn, self).__init__()

        self.num_imgs = num_imgs
        self.device = torch.device(device)

        # CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.fc_conv_1 = nn.Linear(in_features=16 * 11 * 16, out_features=500)
        self.fc_conv_2 = nn.Linear(in_features=500, out_features=250)

        # Estimator
        self.fc_class_1 = nn.Linear(in_features=self.num_imgs * (250 + 2), out_features=500)
        self.fc_class_2 = nn.Linear(in_features=500, out_features=125)
        self.fc_class_3 = nn.Linear(in_features=125, out_features=2)

    def _cnn(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_conv_1(x))
        x = F.relu(self.fc_conv_2(x))
        return x

    def _estimator(self, x):
        x = F.relu(self.fc_class_1(x))
        x = F.relu(self.fc_class_2(x))
        x = self.fc_class_3(x)
        return x

    def forward(self, imgs, actions):
        assert imgs.shape[0] == self.num_imgs, "num_imgs: {}".format(self.num_imgs)
        assert actions.shape[0] == self.num_imgs, "num_imgs: {}".format(self.num_imgs)

        feature_vec = torch.empty(size=(1, self.num_imgs * (250 + 2)), dtype=torch.float, device=self.device)
        for idx, img in enumerate(imgs):
            feature_vec[:, 250 * idx:250*(idx + 1)] = self._cnn(img.unsqueeze(0))
        for idx, action in enumerate(actions):
            feature_vec[:, 250 * self.num_imgs + 2 * idx: 250 * self.num_imgs + 2 * (idx + 1)] = action

        return self._estimator(feature_vec)


# class ActionEstimator(nn.Module):
#
#     def __init__(self):
#         super(ActionEstimator, self).__init__()
#
#         self.cnn = SimpleCNN()
#
#         self.fc1 = nn.Linear(in_features=2048, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=128)
#         self.fc3 = nn.Linear(in_features=128, out_features=2)
#
#     def forward(self, imgs):
#         assert imgs.shape[0] == 2
#
#         x = self.cnn(imgs)
#         x = torch.cat((x[0, :], x[1, :]))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class DiscreteActionNet(nn.Module):

    def __init__(self, num_actions):
        super(DiscreteActionNet, self).__init__()

        self.cnn = SimpleCNN()
        self.fc_final = nn.Linear(in_features=1024, out_features=num_actions)

    def forward(self, img):
        x = self.cnn(img)
        x = F.relu(self.fc_final(x))
        return x

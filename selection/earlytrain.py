from selection.coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from utils.parallel import MyDataParallel
from torchvision import transforms


class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200, fraction_pretrain=1.,**kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device, task_id)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.network = network
        self._device = device
        self.task_id = task_id

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        self.n_pretrain_size = round(self.fraction_pretrain * self.n_train)


    def train(self, epoch, list_of_train_idx, **kwargs):
        """ Train model for one epoch """
        self.before_train()
        self.model.train()

        print('\n=> Training Epoch #%d' % epoch)
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args["selection_batch"],
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)
        #print(len(self.dst_train))
        train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=False, batch_sampler=batch_sampler,
                                                   num_workers=self.args["workers"], pin_memory=True)
        #print(len(train_loader))
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device).long()
            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)["logits"]
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args["selection_batch"])

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        self.model = self.network.to(self._device)

        if self._device == "cpu":
            print("Using CPU.")
        elif self._device is not None:
            torch.cuda.set_device(self._device)
            #self.model = MyDataParallel(self.model, device_ids=0)
        elif torch.cuda.device_count() > 1:
            self.model = MyDataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().to(self._device)
        #self.criterion.__init__()

        # Setup optimizer
        if self.args["selection_optimizer"] == "SGD":
            self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args["selection_lr"],
                                                   momentum=self.args["selection_momentum"],
                                                   weight_decay=self.args["selection_weight_decay"],
                                                   nesterov=self.args["selection_nesterov"])
        elif self.args["selection_optimizer"] == "Adam":
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["selection_lr"],
                                                    weight_decay=self.args["selection_weight_decay"])
        else:
            self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                       lr=self.args["selection_lr"],
                                                                       momentum=self.args["selection_momentum"],
                                                                       weight_decay=self.args["selection_weight_decay"],
                                                                       nesterov=self.args["selection_nesterov"])

        self.before_run()

        for epoch in range(self.epochs):
            list_of_train_idx = np.random.choice(np.arange(self.n_train), self.n_pretrain_size, replace=True)
            self.before_epoch()
            self.train(epoch, list_of_train_idx)
            self.after_epoch()

        return self.finish_run()

    def test(self, epoch):
        self.model.no_grad = True
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.dst_test if self.args.selection_test_fraction == 1. else
                                                  torch.utils.data.Subset(self.dst_test, np.random.choice(
                                                      np.arange(len(self.dst_test)),
                                                      round(len(self.dst_test) * self.args.selection_test_fraction),
                                                      replace=False)),
                                                  batch_size=self.args.selection_batch, shuffle=False,
                                                  num_workers=self.args.workers, pin_memory=True)
        correct = 0.
        total = 0.

        print('\n=> Testing Epoch #%d' % epoch)

        for batch_idx, (input, target) in enumerate(test_loader):
            output = self.model(input.to(self._device))
            loss = self.criterion(output, target.to(self._device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.args.print_freq == 0:
                print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\tTest Loss: %.4f Test Acc: %.3f%%' % (
                    epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
                                                        self.args.selection_batch) + 1, loss.item(),
                    100. * correct / total))

        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

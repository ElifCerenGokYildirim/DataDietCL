import copy

from .earlytrain import EarlyTrain
import torch, time
import numpy as np
#from ..nets.nets_utils import MyDataParallel


class GraNd(EarlyTrain):
    def __init__(self,network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200, repeat=1,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed,device,task_id, epochs, **kwargs)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat
        self._device = device
        self.task_id = task_id

        self.balance = balance

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args["print_freq"] == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        pass

    def finish_run(self):
        self._model = copy.deepcopy(self.model)
        self._model.embedding_recorder.record_embedding = True  # recording embedding vector

        self._model.eval()

        embedding_dim = self._model.get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args["selection_batch"], num_workers=self.args["workers"])
        sample_num = self.n_train

        for i, (_, inputs, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outputs = self._model(inputs)["logits"]
            loss = self.criterion(outputs.requires_grad_(True),targets).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                self.norm_matrix[i * self.args["selection_batch"]:min((i + 1) * self.args["selection_batch"], sample_num),
                self.cur_repeat] = torch.norm(torch.cat([bias_parameters_grads, (
                        self._model.embedding_recorder.embedding["features"].view(batch_num, 1, embedding_dim).repeat(1,
                                             self.end_label, 1) * bias_parameters_grads.view(
                                             batch_num, self.end_label, 1).repeat(1, 1, embedding_dim)).
                                             view(batch_num, -1)], dim=1), dim=1, p=2)

        self._model.train()

        self._model.embedding_recorder.record_embedding = False

    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self._device)
        start_label = self.task_id * 10
        self.end_label = (self.task_id + 1) * 10
        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = self.random_seed + 5

        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(start_label, self.end_label):
                c_indx = self.train_indx[self.dst_train.labels == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": self.norm_mean}

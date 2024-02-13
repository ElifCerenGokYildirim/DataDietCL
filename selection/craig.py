import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from .earlytrain import EarlyTrain
import torch
from .selection_utils import FacilityLocation, submodular_optimizer
import numpy as np
from .selection_utils.euclidean import euclidean_dist_pair_np
#from ..nets.nets_utils import MyDataParallel


class Craig(EarlyTrain):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200, specific_model=None,
                 balance=True, greedy="LazyGreedy", **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device, task_id, epochs, **kwargs)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy
        self.balance = balance
        self.task_id = task_id
        self._device = device

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args["print_freq"] == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def calc_gradient(self, index=None):
        self._model.eval()

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args["selection_batch"], num_workers=self.args["workers"])

        sample_num = len(self.dst_val.targets) if index is None else len(index)

        self.embedding_dim = self._model.get_last_layer().in_features

        gradients = []

        for i, (_,inputs, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outputs = self._model(inputs)["logits"]
            loss = self.criterion(outputs.requires_grad_(True),targets).sum()
            batch_num = targets.shape[0]

            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self._model.embedding_recorder.embedding["features"].view(batch_num, 1,
                                                                                       self.embedding_dim).repeat(1,
                                                                                                                  self.end_label,
                                                                                                                  1) * bias_parameters_grads.view(
                    batch_num, self.end_label, 1).repeat(1, 1, self.embedding_dim)
                gradients.append(
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0)

        self._model.train()
        return euclidean_dist_pair_np(gradients)

    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

    def finish_run(self):
        self._model = copy.deepcopy(self.model)
        self._model.no_grad = True
        with self._model.embedding_recorder:
            if self.balance:
                # Do selection by class
                selection_result = np.array([], dtype=np.int32)
                weights = np.array([])
                start_label = self.task_id * 10
                self.end_label = (self.task_id + 1) * 10
                for c in range(start_label, self.end_label):
                    class_index = np.arange(self.n_train)[self.dst_train.labels == c]
                    matrix = -1. * self.calc_gradient(class_index)
                    matrix -= np.min(matrix) - 1e-3
                    submod_function = FacilityLocation(index=class_index, similarity_matrix=matrix)
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=class_index,
                                                                                   budget=round(self.fraction * len(
                                                                                       class_index)))
                    class_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)
                    selection_result = np.append(selection_result, class_result)
                    weights = np.append(weights, self.calc_weights(matrix, np.isin(class_index, class_result)))
            else:
                matrix = np.zeros([self.n_train, self.n_train])
                all_index = np.arange(self.n_train)
                for c in range(self.args.num_classes):  # Sparse Matrix
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    matrix[np.ix_(class_index, class_index)] = -1. * self.calc_gradient(class_index)
                    matrix[np.ix_(class_index, class_index)] -= np.min(matrix[np.ix_(class_index, class_index)]) - 1e-3
                submod_function = FacilityLocation(index=all_index, similarity_matrix=matrix)
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=all_index,
                                                                               budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain_batch,
                                                           update_state=submod_function.update_state,
                                                           batch=self.args.selection_batch)
                weights = self.calc_weights(matrix, selection_result)
        self._model.no_grad = False
        return {"indices": selection_result, "weights": weights}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

import copy

from .earlytrain import EarlyTrain
import numpy as np
import torch
from .selection_utils import cossim_np, submodular_function, submodular_optimizer
#from ..nets.nets_utils import MyDataParallel


class Submodular(EarlyTrain):
    def __init__(self, network,dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200, balance=True,
                 function="LogDeterminant", greedy="ApproximateLazyGreedy", metric="cossim", **kwargs):
        super(Submodular, self).__init__(network, dst_train, args, fraction, random_seed, device, task_id, epochs, **kwargs)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._device = device
        self._greedy = greedy
        self._metric = metric
        self._function = function
        self.task_id = task_id
        self.balance = balance

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

    def calc_gradient(self, end_label, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self._model.eval()

        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args["selection_batch"],
                num_workers=self.args["workers"])
        sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self._model.get_last_layer().in_features
        print("embedding dimension", self.embedding_dim)

        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []

        for i, (_, inputs, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            outputs = self._model(inputs)["logits"]
            loss = self.criterion(outputs.requires_grad_(True), targets).sum()
            batch_num = targets.shape[0]

            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]

                weight_parameters_grads = self._model.embedding_recorder.embedding["features"].view(batch_num, 1,
                                                                                       self.embedding_dim).repeat(1,
                                                                                                                  end_label,
                                                                                                                  1) * \
                                          bias_parameters_grads.view(batch_num, end_label,
                                                                     1).repeat(1, 1, self.embedding_dim)

                print("TICK")
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1).cpu().numpy())
                print(len(gradients))

        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def finish_run(self):

        # Turn on the embedding recorder and the no_grad flag
        with self.model.embedding_recorder:
            self._model = copy.deepcopy(self.model)
            self._model.no_grad = True
            self.train_indx = np.arange(self.n_train)

            if self.balance:
                selection_result = np.array([], dtype=np.int64)
                print("current task id", self.task_id)
                start_label = self.task_id * 10
                end_label = (self.task_id + 1) * 10
                for c in range(start_label, end_label):
                    c_indx = self.train_indx[self.dst_train.labels == c]
                    print("c index", c_indx)
                    # Calculate gradients into a matrix
                    gradients = self.calc_gradient(end_label,index=c_indx)
                    # Instantiate a submodular function
                    submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                        similarity_kernel=lambda a, b:cossim_np(gradients[a], gradients[b]))
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args,
                                        index=c_indx, budget=round(self.fraction * len(c_indx)), already_selected=[])

                    c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                                 update_state=submod_function.update_state)
                    selection_result = np.append(selection_result, c_selection_result)
            else:
                # Calculate gradients into a matrix
                gradients = self.calc_gradient()
                # Instantiate a submodular function
                submod_function = submodular_function.__dict__[self._function](index=self.train_indx,
                                            similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=self.train_indx,
                                                                                  budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)

            self.model.no_grad = False
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()

        return selection_result



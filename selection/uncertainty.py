from .earlytrain import EarlyTrain
import torch
import numpy as np


class Uncertainty(EarlyTrain):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200, selection_method="LeastConfidence",
                 specific_model=None, balance=False, **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device, task_id, epochs, **kwargs)
        self._device = device
        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method
        print("Selection", self.selection_method)

        self.epochs = epochs
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

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            start_label = self.task_id * 10
            end_label = (self.task_id + 1) * 10
            scores = []
            for c in range(start_label, end_label):
                class_index = np.arange(self.n_train)[self.dst_train.labels == c]
                scores.append(self.rank_uncertainty(class_index))
                selection_result = np.append(selection_result, class_index[np.argsort(scores[-1])[
                                                               :round(len(class_index) * self.fraction)]])
        else:
            scores = self.rank_uncertainty()
            selection_result = np.argsort(scores)[::-1][:self.coreset_size]
        return {"indices": selection_result, "scores": scores}

    def rank_uncertainty(self, index):
        self.model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args["selection_batch"],
                num_workers=self.args["workers"],drop_last=False)
            scores = np.array([])
            batch_num = len(train_loader)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                logits = self.model(inputs)["logits"]
                logits = logits.to(self._device)
                if i % self.args["print_freq"] == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, logits.max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                    new_scores = (np.log(preds + 1e-6) * preds).sum(axis=1)
                    scores = np.append(scores, new_scores)
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(logits, dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

class CoresetMethod(object):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            print("fraction", fraction)
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.num_classes = len(dst_train.labels)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

    def select(self, **kwargs):
        return
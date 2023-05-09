
class EarlyStopping:
    """ Early stopping for training if validation loss doesn't improve for specific number"""
    def __init__(self, patience=5):
        """
            patience: this is specific number before model stop if there's no improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stopping = False

    def __call__(self, val_loss):

        score = val_loss
        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score:
            self.counter += 1
            self.early_stopping = True

        else:
            self.best_score = score
            self.counter = 0


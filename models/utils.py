

class EarlyStopping:
    def __init__(self, patience=20, conv_threshold=1e-6):
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_accuracy = 0
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after loss stops improving to stop
        self.conv_threshold = conv_threshold
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, train_loss, val_loss, accuracy):
        if ((train_loss + self.conv_threshold) <= self.best_train_loss) and (val_loss <= self.best_val_loss or accuracy > self.best_accuracy): # best epoch criterion
            self.best_epoch = epoch
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.best_accuracy = accuracy
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(f'\nStopping training early as no improvement observed in last {self.patience} epochs.',
                f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n')
        return stop

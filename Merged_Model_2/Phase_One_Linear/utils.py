import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None

    def step(self, acc, model, epoch, print_less):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)

        elif score < self.best_score:
            self.counter += 1
            if not print_less : print(f'Patience : {self.counter} / {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        
        else:
            if not print_less : print()
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

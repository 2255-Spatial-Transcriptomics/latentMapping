from models.descriminatorModels.classifier import BinaryClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class BinaryClassifierTrainer(): 
    def __init__(self, n_latent):
        
        self.model = BinaryClassifier(n_latent)
        self.lr = 0.001
        self.weight_decay = 0
        self.batch_size = 10
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                          lr=self.lr, weight_decay=self.weight_decay)

        
    def get_acc(self, pred, true):
        pred_vals = torch.round(torch.sigmoid(pred))

        correct_results_sum = (pred_vals == true).sum().float()
        acc = correct_results_sum/true.shape[0]
        acc = torch.round(acc * 100)
        
        return acc

    def train(self, zs, labels):
        dataset = TensorDataset(zs, labels)
        dataloader = DataLoader(dataset, batch_size = self.batch_size)
        
        batch_loss = 0
        batch_acc = 0 
        for z_batch, labels_true in iter(dataloader):
            self.optimizer.zero_grad()
            labels_pred = self.model(z_batch)
            
            loss = self.loss_function(labels_pred, labels_true)
            acc = self.get_acc(labels_pred, labels_true) 
            loss.backward()
            self.optimizer.step()
            batch_loss += loss.item()
            batch_acc += acc.item()
        
        total_loss = batch_loss / len(dataloader)
        total_acc = batch_acc / len(dataloader)
        return total_loss, total_acc

    def forward(self, Z):
        return self.model(Z)
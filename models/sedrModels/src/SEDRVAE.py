#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )
    
class AddLayer(nn.Module):
    def __init__(self, value):
        super(AddLayer, self).__init__()
        self.value = value

    def forward(self, x):
        return x + self.value
    


class SEDRVAE(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDRVAE, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.latent_dim
        
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.fc_mu = nn.Sequential()
        self.fc_mu.add_module('fc_mu', nn.Linear(params.feat_hidden2, self.latent_dim))
        
        self.fc_logvar = nn.Sequential()
        self.fc_logvar.add_module('fc_logvar', nn.Linear(params.feat_hidden2, self.latent_dim))
        self.fc_logvar.add_module('relu', nn.ReLU())
        self.fc_logvar.add_module('add1', AddLayer(1))
        
        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', nn.Linear(self.latent_dim, input_dim))
        # self.decoder.add_module('tanh', nn.Tanh())
        
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, self.latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x):
        feat_x = self.encoder(x)
        mu = self.fc_mu(feat_x)
        logvar = self.fc_logvar(feat_x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # breakpoint()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        de_feat = self.decoder(z)

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q
    

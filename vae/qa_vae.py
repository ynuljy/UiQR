import torch
import torch.nn as nn
import torch.nn.functional as F

class Qa_VAE(nn.Module):
    def __init__(self, entity_dim, relation_dim, latent_dim, pretrained_en_embeds,
                 pretrained_re_embeddings, k_hop, num_heads=2, num_layers=6):
        super(Qa_VAE, self).__init__()

        self.en_dim = entity_dim * (2 * k_hop)
        self.relation_dim = relation_dim
        self.latent_dim = latent_dim
        self.pretrained_en_embeds = pretrained_en_embeds
        self.pretrained_relation_embeddings = pretrained_re_embeddings

        self.lstm = nn.LSTM(input_size=self.en_dim, hidden_size=self.en_dim, batch_first=True)

        self.fc_mu = nn.Linear(self.en_dim, latent_dim)
        self.fc_var = nn.Linear(self.en_dim, latent_dim)

        nn.init.kaiming_uniform_(self.fc_mu.weight)
        nn.init.kaiming_uniform_(self.fc_var.weight)

        relation_num = pretrained_re_embeddings.weight.shape[0]

        # decoder
        self.fc1 = nn.Linear(latent_dim + relation_num, 256)
        self.fc_out = nn.Linear(256, self.relation_dim)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc_out.weight)


    def encode(self, en_pair_emb):
        mu = self.fc_mu(en_pair_emb)
        log_var = self.fc_var(en_pair_emb)

        return mu, log_var

    def decode(self, z, neighbor_rels_onehot):
        z = torch.cat((z, neighbor_rels_onehot), dim=-1)
        layer1 = self.fc1(z)
        layer1 =torch.relu(layer1)
        output = self.fc_out(layer1)
        return output

    def forward(self, emb_set, neighbor_rels_onehot):
        emb_set = torch.relu(emb_set)
        if emb_set.shape[-1] < self.en_dim:
            padding_size = self.en_dim - emb_set.shape[-1]
            padded_input = F.pad(emb_set, (0, padding_size))  
            lstm_out, (h_n, c_n) = self.lstm(padded_input.unsqueeze(1))  # [batch_size, 1, new_emb_dim]
        else:
            lstm_out, (h_n, c_n) = self.lstm(emb_set.unsqueeze(1))  # [batch_size, 1, new_emb_dim]

        h_last = h_n[-1, :, :]

        mu, log_var = self.encode(h_last)
        std = torch.exp(0.5 * log_var)

        z = mu + std * torch.randn_like(mu)
        pathes_embeddings = self.decode(z, neighbor_rels_onehot)

        return pathes_embeddings, mu, log_var


    def loss_function(self, path_prediction, p_rels_groundtruth, mu, log_var, epoch, total_epoch):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        sim_loss = 1 - F.cosine_similarity(path_prediction, p_rels_groundtruth, dim=-1)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()), dim=0)
        kld_weight = 1
        loss = kld_weight * kld_loss + sim_loss
        return loss.mean()

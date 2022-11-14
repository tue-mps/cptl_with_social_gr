import torch
from torch.nn import functional as F
import torch.nn as nn
from generative_model.linear_nets import MLP,fc_layer,fc_layer_split
from helper.replayer import Replayer
from torch.autograd import Variable
import numpy as np

from helper.utils import l2_loss


class AutoEncoder(Replayer):
    '''Class for variational auto-encoder (VAE) models.'''

    def __init__(self, x_dim=2, h_dim=64, z_dim=16, n_layers=1, sample_k=20, pred_len=8, bias=False):
        super().__init__()
        self.label = "VRNN"
        self.average = "average"

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.sample_k = sample_k
        self.pred_len = pred_len

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.enc_mean = nn.Linear(h_dim, z_dim)

        self.enc_logvar = nn.Linear(h_dim, z_dim)  # nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.prior_mean = nn.Linear(h_dim, z_dim)

        self.prior_logvar = nn.Linear(h_dim, z_dim)  # nn.Softplus()

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.dec_logvar = nn.Linear(h_dim, x_dim)  # nn.Softplus()

        self.dec_mean = nn.Sequential(nn.Linear(self.h_dim, self.x_dim),
                                      nn.Hardtanh(min_val=-10, max_val=10))  # nn.Sigmoid()

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        # self.l_abs = nn.Linear(self.x_dim, self.h_dim)

    @property
    def name(self):
        return "{}".format("Generator --> VAE")

    # --------------------------------------------------------------------------------------#
    ##---- FORWARD FUNCTIONS ----##
    # --------------------------------------------------------------------------------------#

    def _encoder(self, phi_x_t, h):
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_logvar_t = self.enc_logvar(enc_t)
        return enc_mean_t, enc_logvar_t

    def _prior(self, h):
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        prior_logvar_t = self.prior_logvar(prior_t)
        return prior_mean_t, prior_logvar_t

    def _decoder(self, phi_z_t, h):
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        dec_mean_t = self.dec_mean(dec_t)
        dec_logvar_t = self.dec_logvar(dec_t)
        return dec_mean_t, dec_logvar_t

    def forward(self, x, obs_traj_in):
        """
        Inputs:
        - x: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        kld_loss, nll_loss = 0, 0
        x_list, mean_list = [torch.zeros(2)], [torch.zeros(2)]

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim), requires_grad=True).cuda()
        # h = self.l_abs(obs_traj_in.cuda()).unsqueeze(0)

        for t in range(1, x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder mean and logvar
            enc_mean_t, enc_logvar_t = self._encoder(phi_x_t, h)

            # prior mean and logvar
            prior_mean_t, prior_logvar_t = self._prior(h)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_logvar_t)
            phi_z_t = self.phi_z(z_t.cuda())

            # decoder
            dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_logvar_t, x[t])

            """
            self.writer.add_histogram('input_trajectory', x[t], t)
            self.writer.add_histogram('decoder_mean', dec_mean_t, t)
            """

            x_list.append(x[t][0])
            mean_list.append(dec_mean_t[0])

        return kld_loss, nll_loss, h

    ##------- SAMPLE FUNCTIONS -------##

    def _generate_sample(self, h):
        # prior mean and logvar
        prior_mean_t, prior_logvar_t = self._prior(h)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t.cuda())

        # decoder
        dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)

        sample_t = self._reparameterized_sample(dec_mean_t, dec_logvar_t)

        return sample_t, phi_z_t

    def sample(self, seq_len, batch_dim, h_prec=None):
        with torch.no_grad():
            if h_prec is None:
                h = Variable(torch.zeros(self.n_layers, batch_dim, self.h_dim)).cuda()
                sample = torch.zeros(seq_len, batch_dim, self.x_dim)

                for t in range(seq_len):
                    sample_t, phi_z_t = self._generate_sample(h)
                    phi_x_t = self.phi_x(sample_t.cuda())
                    sample[t] = sample_t.data
                    # recurrence
                    _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            else:
                h = h_prec
                sample = torch.zeros(seq_len, batch_dim, self.x_dim)

                for t in range(seq_len):
                    sample_t, phi_z_t = self._generate_sample(h)
                    phi_x_t = self.phi_x(sample_t.cuda())
                    sample[t] = sample_t.data
                    # recurrence
                    _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

        return sample

    ##-------- LOSS FUNCTIONS --------##

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, logvar):
        """Using std to sample"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).cuda()
        return mean + eps * std

    def _kld_gauss(self, mean_enc, logvar_enc, mean_prior, logvar_prior):
        """Using std to compute KLD"""
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(self, mean, logvar, x):
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll

    def calculate_recon_loss(self, x, x_recon, mode=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]         <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]   (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]   <bool>, if True, loss is average over all frames; otherwise it is summed

        OUTPUT: - [reconL]    <1D-tensor> of length [batch_size]
        '''
        # x = x.permute(1,0,2)
        # x_recon = x_recon.permute(1,0,2)
        # batch_size = x.size(0)
        seq_len, batch, size = x.size()
        # reconL = F.binary_cross_entropy(input=x_recon.reshape(batch_size, -1), target=x.reshape(batch_size, -1),
        #                                 reduction='none')
        # reconL = torch.mean(reconL, dim=1) if mode else torch.sum(reconL, dim=1)

        reconL = (x.permute(1, 0, 2) - x_recon.permute(1, 0, 2)) ** 2
        if mode == "sum":
            return torch.sum(reconL)
        elif mode == "average":
            return torch.sum(reconL) / (batch * seq_len * size)
        elif mode == "raw":
            return reconL.sum(dim=2).sum(dim=1)

    def calculate_variat_loss(self, mu, logvar):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [mu]      <2D-tensor> by encoder predicted mean for [z]
                - [logvar]  <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL] <1D-tensor> of length [batch_size]
        '''

        # --> calculate analytically
        # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
        variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return variatL



    def loss_function(self, recon_x, x, y_hat=None, y_target=None, scores=None, mu=None, logvar=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:   - [recon_x]     <4D-tensor> reconstructed traj in same shape as [x]
                 - [x]           <4D-tensor> original traj
                 - [y_hat]       <2D-tensor> predicted traj
                 - [y_target]    <2D-tensor> future traj
                 - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
                 - [logvar]      None or <2D-tensor> with estimated log(SD^2) of [z]

        SETTING: - [self.average] <bool>, if True, both [reconL] and [variatL] are divided by number of input elements

        OUTPUT:  - [reconL]      reconstruction loss indicating how well [x] and [x_recon] match
                 - [variatL]     variational (KL-divergence) loss "indicating how normally distributed [z] is"
                 - [predL]       prediction loss indicating how well targets [y] are predicted
                 - [distilL]     knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                    match the target "logits" ([scores])
        '''

        ###---Reconstruction loss---###
        reconL = self.calculate_recon_loss(x=x, x_recon=recon_x, mode=self.average)  # -> possibly average over traj
        reconL = torch.mean(reconL)  # -> average over batch

        ###--- Variational loss ----###
        if logvar is not None:
            variatL = self.calculate_variat_loss(mu=mu, logvar=logvar)
            variatL = torch.mean(variatL)  # -> average over batch
            if self.average:
                pass  # todo
        else:
            variatL = torch.tensor(0., device=self._device())

        '''
        ###----Prediction loss----###
        if y_target is not None:
            predL = F.cross_entropy(y_hat, y_target, reduction='mean')  #-> average over batch
        else:
            predL = torch.tensor(0., device=self._device())
        '''
        # Return a tuple of the calculated losses
        return reconL, variatL

    ##------- TRAINING FUNCTIONS -------##

    def train_a_batch(self, x, y, seq_start_end, x_=None, y_=None, seq_start_end_=None, rnt=0.5):
        '''Train model for one batch ([x],[y]),possibly supplemented with replayed data ([x_],[y_])

        [x]          <tensor> batch of past trajectory (could be None, in which case only 'replayed' data is used)
        [y]          <tensor> batch of corresponding future trajectory
        [x_]         None or (<list> of) <tensor> batch of replayed past trajectory
        [y_]         None or (<list> of) <tensor> batch of corresponding future trajectory
        [rnt]        <number> in [0,1], relative importance of new task
        '''

        # Set model to training-mode
        self.train()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        if x is not None:
            # Run the model
            kld_loss, variat_loss, h = self(x, x[0])

            v_losses = []
            for i in range(0, self.sample_k):
                pred_traj_rel= self.sample(self.pred_len, x.size(1), h).cuda()
                reconL = self.calculate_recon_loss(x=x, x_recon=pred_traj_rel, mode=self.average)
                v_losses.append(reconL)
            reconL_min = min(v_losses)
            loss_cur = kld_loss + variat_loss + reconL_min

            # Calculate training-precision  #

        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:


            n_replays = 1

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays

            kld_loss_r = [None]*n_replays
            variat_loss_r = [None]*n_replays
            reconL_min_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task)

            # Loop to perform each replay
            for replay_id in range(n_replays):

                if (not type(x_) == list):
                    x_temp_ = x_
                    kld_loss_r[replay_id], variat_loss_r[replay_id], h = self(x_temp_, x_temp_[0])

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_) == list):
                    x_temp_ = x_[replay_id]
                    kld_loss_r[replay_id], variat_loss_r[replay_id], h = self(x_temp_, x_temp_[0])
                v_losses = []
                for i in range(0, self.sample_k):
                    pred_traj_rel = self.sample(self.pred_len, x_temp_.size(1), h).cuda()
                    reconL = self.calculate_recon_loss(x=x_temp_, x_recon=pred_traj_rel, mode=self.average)
                    v_losses.append(reconL)
                reconL_min_r[replay_id] = min(v_losses)

                # Weigh losses as requested
                loss_replay[replay_id] = kld_loss_r[replay_id] + variat_loss_r[replay_id] + reconL_min_r[replay_id]
                '''
                if self.replay_target=="hard":
                    loss_replay[replay_id] += self.lamda_pl*predL_r[replay_id]
                elif self.replay_target=="soft":
                    loss_replay[replay_id] += self.lamda_pl*predL_r[replay_id]
                '''

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays
        loss_total = loss_replay if (x is None) else (
            loss_cur if x_ is None else rnt * loss_cur + (1 - rnt) * loss_replay)

        # Reset optimizer
        self.optimizer.zero_grad()

        # Backpropagate errors
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'reconL': reconL_min.item() if x is not None else 0,
            'variatL': variat_loss.item() if x is not None else 0,
            # 'predL': predL.item() if x is not None else 0,
            'reconL_r': sum(reconL_min_r).item()/n_replays if x_ is not None else 0,
            'variatL_r': sum(variat_loss_r).item()/n_replays if x_ is not None else 0,
            # 'predL_r': sum(predL_r).item()/n_replays if x_ is not None else 0,
        }

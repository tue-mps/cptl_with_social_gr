import torch
from torch.nn import functional as F
import torch.nn as nn


from exemplars import ExemplarHandler
from replayer import Replayer
from continual_learner import ContinualLearner


class Predictor(ContinualLearner, Replayer, ExemplarHandler):
    '''Model for predicting trajectory, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    # reference GAN code, generator part, encoder & decoder (LSTM)
    def __init__(
            self,
            obs_len,
            pred_len,
            traj_lstm_input_size,
            traj_lstm_hidden_size,
            traj_lstm_output_size,
            dropout,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_output_size = traj_lstm_output_size

        #--------------------------MAIN SPECIFY MODEL------------------------#

        #-------Encoder-------#
        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)

        #-------Decoder------#
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_hidden_size, traj_lstm_output_size)
        self.pred_hidden2pos =nn.Linear(self.traj_lstm_output_size, 2)

    # initial encoder traj lstm hidden states
    def init_encoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )
    # initial decoder traj lstm hidden states
    def init_decoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
        )

    def forward(self, obs_traj_rel, obs_traj_pos, seq_start_end):
        batch = obs_traj_pos.shape[1] #todo define the batch
        traj_lstm_h_t, traj_lstm_c_t = self.init_encoder_traj_lstm(batch)
        pred_lstm_h_t, pred_lstm_c_t = self.init_decoder_traj_lstm(batch)
        pred_traj_pos = []
        traj_lstm_hidden_states = []
        pred_lstm_hidden_states = []

        # encoder, calculate the hidden states
        for i, input_t in enumerate(
            obs_traj_pos[: self.obs_len].chunk(
                obs_traj_pos[: self.obs_len].size(0), dim=0
            )
        ):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]

        output = obs_traj_pos[self.obs_len-1]
        if self.training:
            for i, input_t in enumerate(
                obs_traj_pos[-self.pred_len :].chunk(
                    obs_traj_pos[-self.pred_len :].size(0), dim=0
                )
            ):
                pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_h_t, pred_lstm_c_t)
                )
                output = self.pred_hidden2pos(pred_lstm_h_t)
                pred_traj_pos += [output]
            outputs = torch.stack(pred_traj_pos)
        else:
            for i in range(self.pred_len):
                pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                    output, (pred_lstm_h_t, pred_lstm_c_t)
                )
                output = self.pred_hidden2pos(pred_lstm_h_t)
                pred_traj_pos += [output]
            outputs = torch.stack(pred_traj_pos)
        return outputs

    def train_a_batch(self, x, y, x_=None, y_=None, active_classes=None, rnt=0.5):
        '''
        Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_], [y_]).

        [x]       <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]       <tensor> batch of corresponding prediction trajectory
        [x_]      None or (<list> of) <tensor> batch of replayed inputs
        [y_]      None or (<list> of) <tensor> batch of corresponding "replayed"  prediction trajectory

        '''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        #--(1)-- REPLAYED DTA---#

        if x_ is not None:
            y_ = [y_]
            n_replays = len(y_) if (y_ is not None) else None

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            pred_traj_r = [None]*n_replays
            distill_r = [None]*n_replays

            # Loop to evaluate predictions on replay according to each previous task
            y_hat_all = self(x_)

            for replay_id in range(n_replays):
                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classed not in replayed task
                # y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]
                y_hat = y_hat_all

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    pred_traj_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')

                # Weigh losses
                loss_replay[replay_id] = pred_traj_r[replay_id]

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays


        #--(2)-- CURRENT DATA --#

        if x is not None:
            # Run model
            y_hat = self(x)

            # Calculate prediction loss
            pred_traj = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = pred_traj


        # Combine loss from current and replayed batch
        loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay


        #--(3)-- ALLOCATION LOSSES --#

        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Returen the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current':loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred_traj': pred_traj.item() if pred_traj is not None else 0,
            'pred_traj_r': pred_traj_r.item() if pred_traj_r is not None else 0,
        }





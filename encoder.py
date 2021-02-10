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

    def train_a_batch(self):
        pass






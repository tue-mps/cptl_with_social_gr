import torch
from torch.nn import functional as F
import torch.nn as nn

from helper.replayer import Replayer
from helper.continual_learner import ContinualLearner
from helper.utils import l2_loss

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%Ss"' % noise_type)

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data



class Predictor(ContinualLearner, Replayer):
    '''Model for predicting trajectory, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    # reference GAN code, generator part, encoder & decoder (LSTM)
    def __init__(
            self,
            obs_len,
            pred_len,
            traj_lstm_input_size,
            traj_lstm_hidden_size,
            traj_lstm_output_size,
            noise_dim=(8,),
            noise_type="gaussian",
            n_units=None,
            n_heads=None,
            graph_network_out_dims=32,
            dropout=0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
    ):
        super().__init__()
        self.label = "gat"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_output_size = traj_lstm_output_size

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.graph_lstm_hidden_size = graph_lstm_hidden_size


        #--------------------------MAIN SPECIFY MODEL------------------------#

        #-------Encoder-------#
        self.traj_lstm_model_encoder = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)

        self.gatencoder = GATEncoder(n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha)
        self.graph_lstm_model = nn.LSTMCell(graph_network_out_dims, graph_lstm_hidden_size)

        #-------Decoder------#
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_output_size+noise_dim[0]+graph_lstm_hidden_size)
        self.pred_hidden2pos =nn.Linear(self.traj_lstm_output_size+noise_dim[0]+graph_lstm_hidden_size, 2)



    # initial encoder traj lstm hidden states
    def init_encoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )
    # initial decoder traj lstm hidden states
    def init_decoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
        )

    # add noise before decoder
    def add_noise(self, _input):
        noise_shape = (_input.size(0),) + self.noise_dim
        z_decoder = get_noise(noise_shape, self.noise_type)
        decoder_h = torch.cat([_input, z_decoder], dim=1)
        return decoder_h



    # def is_on_cuda(self):
    #     return next(self.parameters()).is_cuda

    @property
    def name(self):
        return "{}".format("gat")

    def forward(self, obs_traj_pos, seq_start_end):
        batch = obs_traj_pos.shape[1] #todo define the batch
        traj_lstm_h_t, traj_lstm_c_t = self.init_encoder_traj_lstm(batch)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch)
        pred_traj_pos = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        # encoder, calculate the hidden states

        for i, input_t in enumerate(
                obs_traj_pos[: self.obs_len].chunk(
                    obs_traj_pos[: self.obs_len].size(0), dim=0
                )
        ):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model_encoder(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]

        graph_lstm_input = self.gatencoder(torch.stack(traj_lstm_hidden_states), seq_start_end)

        for i, input_t in enumerate(
                graph_lstm_input[: self.obs_len].chunk(
                    graph_lstm_input[: self.obs_len].size(0), dim=0
                )
        ):
            graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                input_t.squeeze(0), (graph_lstm_h_t, graph_lstm_c_t)
            )
            graph_lstm_hidden_states += [graph_lstm_h_t]

        pred_lstm_h_t_before_noise = torch.cat((traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1)

        pred_lstm_h_t = self.add_noise(pred_lstm_h_t_before_noise)
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()
        output = obs_traj_pos[self.obs_len-1]

        for i in range(self.pred_len):
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                output, (pred_lstm_h_t, pred_lstm_c_t)
            )
            output = self.pred_hidden2pos(pred_lstm_h_t)
            pred_traj_pos += [output]
        outputs = torch.stack(pred_traj_pos)


        return outputs

    def train_a_batch(self, x_rel, y_rel, seq_start_end, x_rel_=None, y_rel_=None, seq_start_end_=None, loss_mask=None, active_classes=None, rnt=0.5):
        '''
        Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_], [y_]).

        [x]       <tensor> batch of past trajectory (could be None, in which case only 'replayed' data is used)
        [y]       <tensor> batch of corresponding future trajectory
        [x_]      None or (<list> of) <tensor> batch of replayed past trajectory
        [y_]      None or (<list> of) <tensor> batch of corresponding "replayed"  future trajectory

        '''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        #--(1)-- REPLAYED DATA---#

        if x_rel_ is not None:
            y_ = [y_rel_]
            n_replays = len(y_) if (y_ is not None) else None

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            pred_traj_r = [None]*n_replays
            distill_r = [None]*n_replays

            # Loop to evaluate predictions on replay according to each previous task
            y_hat_all = self(x_rel_, seq_start_end_)


            for replay_id in range(n_replays):
                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classed not in replayed task
                # y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]
                y_hat = y_hat_all

                # Calculate losses
                if (y_rel_ is not None) and (y_[replay_id] is not None):
                    # pred_traj_r[replay_id] = F.cross_entropy(y_hat.permute(1,0,2), y_[replay_id].permute(1,0,2), reduction='mean')
                    pred_traj_r[replay_id] = l2_loss(y_hat, y_[replay_id], mode="average")

                # Weigh losses
                loss_replay[replay_id] = pred_traj_r[replay_id]

        # Calculate total replay loss
        loss_replay = None if (x_rel_ is None) else sum(loss_replay) / n_replays


        #--(2)-- CURRENT DATA --#

        if x_rel is not None:
            # Run model
            y_hat_rel = self(x_rel, seq_start_end)
            # relative to absolute
            # y_hat = relative_to_abs(y_hat_rel, )
            # Calculate prediction loss
            # pred_traj = None if y is None else F.cross_entropy(input=y_hat.permute(1,0,2), target=y.permute(1,0,2), reduction='mean')
            pred_traj = None if y_rel is None else l2_loss(y_hat_rel, y_rel, mode="average")
            # a = torch.numel(loss_mask.data)

            # Weigh losses
            loss_cur = pred_traj


        # Combine loss from current and replayed batch
        if x_rel_ is None:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x_rel is None) else rnt*loss_cur+(1-rnt)*loss_replay


        #--(3)-- ALLOCATION LOSSES --#

        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Returen the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current':loss_cur.item() if x_rel is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x_rel is not None) else 0,
            'pred_traj': pred_traj.item() if pred_traj is not None else 0,
            'pred_traj_r': sum(pred_traj_r).item()/n_replays if (x_rel_ is not None and pred_traj_r[0] is not None) else 0,
        }





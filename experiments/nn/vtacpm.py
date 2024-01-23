from torch import nn
import torch


class VTACPredictiveModel(nn.Module):
    """
        Simpler e2e model give, vision_t, touch_t, action_t
        predicts the vision_t, touch_t and action_t at the next timestep
    """

    def __init__(self,
                 vis_ae=None,
                 touch_ae=None,
                 associative_ae=None,
                 predictive_model=None,
                 skip_connection=True
                 ):
        super(VTACPredictiveModel, self).__init__()
        self.vis_ae = vis_ae
        self.touch_ae = touch_ae
        self.associative_ae = associative_ae
        self.predictive_model = predictive_model

        # if action:
        #     self.conv2d = nn.Conv2d(in_channels=4103, out_channels=4096, kernel_size=3, stride=1, padding='same')
        #     self.bn = nn.BatchNorm2d(num_features=4096)
        #     self.relu = nn.ReLU()
        #
        #     torch.nn.init.xavier_uniform_(self.conv2d.weight)
        #     self.conv2d.bias.data.fill_(0.0)

        # self.associative_cortex = parietal_cortex

    def forward(self, *args):
        v_t0, l_t0, action = args

        # encode/project observations
        ev_t0, hv_t0 = self.vis_ae.encode(v_t0, return_hidden=True)
        el_t0, hl_t0 = self.touch_ae.encode(l_t0, return_hidden=True)
        # er_t0 = self.touch_cortex.encode(r_t0)

        # associate vision and touch (es_t0 = current state)
        es_t0 = self.associative_ae.associate(ev_t0, el_t0)
        es_tx = self.predictive_model(es_t0, action) \
            if action is not None \
            else es_t0

        if action is not None and self.predictive_model is not None:
            es_tx = self.predictive_model(es_t0, action)

        # dissociate vision and touch (s_c = current state)
        ev_tx, el_tx = self.associative_ae.dissociate(es_tx)

        # decode (next) observations
        v_tx = self.vis_ae.decode(ev_tx, hv_t0)
        l_tx = self.touch_ae.decode(el_tx, hl_t0)

        return v_tx, l_tx

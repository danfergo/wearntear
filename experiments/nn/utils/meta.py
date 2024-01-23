from functools import reduce
import torch


def load_weights(self, weights):
    if weights is not None:
        print('loaded weights', weights)
        self.load_state_dict(torch.load(weights))


def run_sequentially(module_list, inputs, return_hidden=False, map_fn=None):
    map_fn = map_fn if map_fn is not None else lambda ht, i: ht

    def cb(ht_hs, ith_layer):
        i, layer = ith_layer
        ht = ht_hs[0] if return_hidden else ht_hs

        ht_next = layer(map_fn(ht, i))
        # print('-->', ht.size(), ht_next.size())
        return (ht_next, ht_hs[1] + (ht_next,)) if return_hidden else ht_next

    return reduce(
        cb,
        enumerate(module_list),
        (inputs, ()) if return_hidden else inputs  # initial values (inputs and empty list to collect Hs)
    )

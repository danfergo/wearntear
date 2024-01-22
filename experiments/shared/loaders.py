import os
from os import path
import random

from dfgiatk.experimenter import e, run
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import ImageLoader, ClassificationLabeler, LambdaLoader, NumpyMapsLabeler
import numpy as np
import cv2

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW, CVT_CHW2HWC
from experiments.shared.transform import img_transform


def loader(partition):
    inputs = e.inputs
    outputs = e.outputs
    dataset_path = e.data_path
    datasets = e.datasets

    # samples is a list of tuples
    # (dataset_name, record_name, record_length)
    samples = [(dataset_name, record_name, len(os.listdir(path.join(dataset_path, dataset_name, record_name, 'c'))))
               for dataset_name in datasets
               for record_name in os.listdir(path.join(dataset_path, dataset_name))]

    def img_path(modality, offset):
        def _(key, stack_rel_offset):
            # key is a tuple
            dataset_name, record_name, random_frame = key

            p = path.join(dataset_path,
                          dataset_name,
                          record_name,  # rec_xxxxx
                          modality,  # l (left touch sensor), r (right touch sensor), c (vision camera)
                          f'frame_{str(key[2] + offset + stack_rel_offset).zfill(5)}.jpg') # sampled frame

            if not os.path.exists(p):
                raise FileNotFoundError()
            return p

        return _

    # def encoded_vector_path(modality, offset):
    #     def _(key):
    #         return path.join(dataset_path, modality, f'{str(key + offset)}.npy')
    #
    #     return _

    positions = {d: {} for d in datasets}
    # positions is a dict
    # dataset_name / record_name / np.array(record_length, 7)
    for s in samples:
        positions_path = path.join(dataset_path, s[0], s[1], 'p.npy')
        with open(positions_path, 'rb') as f:
            positions[s[0]][s[1]] = np.load(f).reshape(-1, 7).astype(np.float32)

    def load_position(key, _):
        # key is a tuple
        dataset_name, record_name, random_frame = key

        pt = positions[dataset_name][record_name][random_frame]
        pt1 = positions[dataset_name][record_name][random_frame + 1]
        return pt1 - pt

    loaders = {
        'l': lambda offset_length, endpoint: ImageLoader(
            transform=img_transform(partition, endpoint, 'l'),
            img_path=img_path('l', offset_length[0]),
            stack=offset_length[1] if offset_length[1] > -1 else False
        ),
        'r': lambda offset_length, endpoint: ImageLoader(
            transform=img_transform(partition, endpoint, 'r'),
            img_path=img_path('r', offset_length[0]),
            stack=offset_length[1] if offset_length[1] > -1 else False
        ),
        'c': lambda offset_length, endpoint: ImageLoader(
            transform=img_transform(partition, endpoint, 'c'),
            img_path=img_path('c', offset_length[0]),
            stack=offset_length[1] if offset_length[1] > -1 else False
        ),
        'a': lambda offset, endpoint: LambdaLoader(load_position)
    }

    def get_loader(k, endpoint):
        # returns the loader for a given endpoint of the model
        # parses the inputs/outputs strings e.g. d:t-l
        # where
        # d - is the data modality,
        # t - offset wrt the sampled timestep
        # l - is the stack of frames length

        key, offset_length = tuple(k.split(':'))
        offset_length = offset_length.split('-') if '-' in offset_length else [offset_length, str(-1)]
        offset_length = [int(x) for x in offset_length]

        return loaders[key](offset_length, endpoint)

    def transform_key(key):
        # key is the randomly chosen index, between 0 and len(samples)
        # samples is a list of tuples: (dataset_name, record_name, record_length)

        # samples a random timestep (t) from the whole video record
        # the timestep needs to be
        rec_sample = samples[key]  # the record
        # todo pass stack length here.
        rnd_frame = random.randint(0, rec_sample[2] - 5 - 2)  # the sampled frame

        # the key becomes a tuple of
        # dataset_name, record_name, random_frame
        return rec_sample[0], rec_sample[1], rnd_frame

    return DatasetSampler(
        samples=np.array(list(range(len(samples)))),
        transform_key=transform_key,
        loader=get_loader(inputs[0], 'i') if len(inputs) == 1 else [get_loader(inp, 'i') for inp in inputs],
        labeler=get_loader(outputs[0], 'o') if len(outputs) == 1 else [get_loader(out, 'o') for out in outputs],
        epoch_size=None if ('random_sampling' in e and not e['random_sampling']) else
        (e.batches_per_epoch if partition == 'train' else e.n_val_batches),
        batch_size=e.batch_size,
        device=e.train_device,
        random_sampling=e.random_sampling if 'random_sampling' in e else True,
        return_names=e.return_names if 'return_names' in e else False,
        self_reset_cache=True
    )


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


    def test_loader():
        loader_ = loader('train')
        for inputs, outputs in loader_:
            x_np = cvt_batch(inputs.cpu().numpy(), CVT_CHW2HWC)
            y_np = cvt_batch(outputs.cpu().numpy(), CVT_CHW2HWC)
            n_samples = x_np.shape[0]
            for i in range(n_samples):
                cv2.imshow('frames', np.concatenate([
                    cv2.cvtColor(x_np[i], cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(y_np[i], cv2.COLOR_RGB2BGR)
                ], axis=1))
                cv2.waitKey(-1)


    run(
        description='Just testing the loaders',
        config={
            'batch_size': 32,
            'batches_per_epoch': 10,
            'train_device': 'cuda',
            'data_path': path.join(__location__, '../../data/'),
            'dataset_name': 'pushing_tower',
        },
        tmp=True,
        entry=test_loader
    )

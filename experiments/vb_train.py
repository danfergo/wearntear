import os

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, ModelSaver
from torch import optim

from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset
from experiments.losses.perceptual_loss import VGGPerceptualLoss
from experiments.shared.loaders import loader

from piqa import SSIM, PSNR, LPIPS

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {
    'description': """
        # VW Vitac PM 
    """,
    'config': {
        'lr': 0.01,

        # data
        'datasets': [
            # 'sim_wear',
            # 'sim_tear',
            # 'real_wear',
            # 'real_tear'
        ],
        'data_path': '/home/danfergo/data/',
        'img_size': (224, 224),
        'o_dt': 4,  # 0 or 1
        '{inputs}': lambda: [f'c:0-{e.o_dt}', f'l:0-{e.o_dt}', f'a:{e.o_dt}'],  # 0-{e.stack}
        '{outputs}': lambda: [f'c:{e.o_dt}', f'l:{e.o_dt}'],
        '{data_loader}': lambda: loader('train'),
        '{val_loader}': lambda: loader('val'),
        # 'train_ic_transform': transform(),

        # network
        # '{model}': lambda: ,

        # train
        'train_device': 'cuda',
        '{perceptual_loss}': lambda: VGGPerceptualLoss().to(e.train_device),
        '{loss}': lambda: (e.perceptual_loss, e.perceptual_loss),
        '{optimizer}': lambda: optim.Adadelta(e.model.parameters(), lr=e.lr),
        'epochs': 2000,
        'batch_size': 32,
        'batches_per_epoch': 50,
        'feed_size': 8,

        # validation
        '{metrics}': lambda: [
            SSIM().to(e.train_device),
            PSNR().to(e.train_device),
            LPIPS().to(e.train_device)
            # HaarPSI().to(e.train_device),
            # VSI().to(e.train_device)
        ],
        # 'HaarPSI', 'VSI'
        'metrics_names': ['SSIM', 'PSNR', 'LPIPS'],
        'n_val_batches': 3,
        'val_feed_size': 8,
    }
}

run(
    **config,
    entry=fit_to_dataset,
    listeners=lambda: [
        Validator(),
        ModelSaver(),
        Logger(),
        Plotter(),
        TrainingSamples(loaders=[
            ('train', e.data_loader, 4),
        ])
    ],
    open_e=False,
    src='experiments'
)

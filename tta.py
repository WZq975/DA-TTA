import os
import logging
import numpy as np

from models.model import get_model
from utils import get_accuracy
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, adaptation_method_lookup

from methods.tent import Tent
from methods.memo import MEMO
from methods.cotta import CoTTA
from methods.rmt import RMT
from methods.eata import EATA
from methods.norm import Norm
from methods.lame import LAME
from methods.sar import SAR
from methods.rotta import RoTTA
from methods.datta import DATTA
import torch
import torchvision

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["fully_tta_noniid",
                      "continual_tta_noniid",
                      ]
    assert cfg.SETTING in valid_settings, f"Choose a setting from: {valid_settings}"
    if cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_r", "imagenet_d"]:
        ALPHA_DIRICHLET = 0.1
    elif cfg.CORRUPTION.DATASET in ["imagenet_c"]:
        ALPHA_DIRICHLET = 0

    base_model = get_model(cfg).cuda()

    # remove the image normalization layer of Hendrycks2020AugMix_ResNeXt, aligning with the "Standard" model for cifar10_c
    # not necessary
    if cfg.MODEL.ADAPTATION == 'datta' and cfg.MODEL.ARCH == 'Hendrycks2020AugMix_ResNeXt':
        for name, buffer in base_model.named_buffers():
            if name == "mu":
                buffer.data = torch.tensor([0.0] * 3).view(1, 3, 1, 1).cuda()
            elif name == "sigma":
                buffer.data = torch.tensor([1.0] * 3).view(1, 3, 1, 1).cuda()

    # setup test-time adaptation method
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"imagenet_d"} and not cfg.CORRUPTION.TYPE[0]:
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    errs = []
    domain_dict = {}
    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_all):
        if i_dom == 0 or "fully_tta_noniid" in cfg.SETTING:
            model.reset()
            logger.info("resetting model")
        else:
            logger.warning("not resetting model")
        for severity in cfg.CORRUPTION.SEVERITY:

            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               domain_names_all=dom_names_all,
                                               alpha_dirichlet=ALPHA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            acc, domain_dict = get_accuracy(model,
                                            data_loader=test_data_loader,
                                            dataset_name=cfg.CORRUPTION.DATASET,
                                            domain_name=domain_name,
                                            setting=cfg.SETTING,
                                            domain_dict=domain_dict)

            err = 1. - acc
            errs.append(err)
            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{cfg.CORRUPTION.SEVERITY}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")

    logger.info(f"mean error: {np.mean(errs):.2%}.")


if __name__ == '__main__':
    evaluate('"Evaluation.')


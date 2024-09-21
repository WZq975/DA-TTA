import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
# from batch_norm import BatchNorm
import json
import os
import tqdm
import PIL
import torchvision.transforms as transforms


from collections import Counter
import torch.nn.functional as F
from methods.base import TTAMethod
from datasets.data_loading import get_source_loader

logger = logging.getLogger(__name__)


def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    means = torch.mean(Y.clone(), dim=(2, 3))
    vars = torch.mean((Y.clone() - means.unsqueeze(2).unsqueeze(3)) ** 2, dim=(2, 3))

    return Y, (means, vars)


class MyBatchNorm(nn.Module):

    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5):
        super().__init__()

        try:
            num_features = bn_init.num_features
        except AttributeError:
            num_features = bn_init.num_feature

        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.weight = nn.Parameter(bn_init.weight.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.eps = 1e-5
        self.register_buffer("weight_init", bn_init.weight.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("bias_init", bn_init.bias.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("mu", torch.ones(1, num_features, 1, 1))
        self.register_buffer("sigma", torch.zeros(1, num_features, 1, 1))

        self.stat = None
        self.reset_mean = True
        self.alpha = datta_alpha

    def reset_statistic(self):
        self.reset_mean = True
        # self.weight.data = self.weight_init.clone().detach()
        # self.bias.data = self.bias_init.clone().detach()

    def set_alpha(self, alpha):
        self.reset_mean = True
        self.alpha = alpha

    def forward(self, X):
        if self.reset_mean:
            self.reset_mean = False
            alpha = self.alpha
            self.mu.data = (1 - alpha) * torch.mean(X, dim=(0, 2, 3), keepdim=True).clone() + alpha * self.running_mean.data.clone()
            self.sigma.data = (1 - alpha) * torch.mean((X - self.mu) ** 2, dim=(0, 2, 3), keepdim=True).clone() + alpha * self.running_var.data.clone()

        Y, self.stat = batch_norm(self.mu, self.sigma, X, self.weight, self.bias, eps=self.eps)

        return Y


class DATTA(TTAMethod):

    def __init__(self, cfg, model, num_classes, steps=1):
        self.alpha = cfg.DATTA.ALPHA  # Equation 6,7
        self.theta = cfg.DATTA.THETA  # Equation 8
        super().__init__(cfg, model, num_classes)
        self.model = model
        self.steps = steps

        self.setting = cfg.SETTING
        self.stat_outputs = {}
        self.loss_list = []
        self.window = []
        self.counter = 0
        self.signal = False
        self.prediction_container = None

        self.W_s = round(cfg.DATTA.WINDOW_SHORT * 40 / cfg.TEST.BATCH_SIZE)
        self.W_l = round(cfg.DATTA.WINDOW_LONG * 40 / cfg.TEST.BATCH_SIZE)
        self.tau = cfg.DATTA.TAU

        dir_source_distri = "./ckpt/source_distribution/"
        self.path_source_distri = dir_source_distri + cfg.CORRUPTION.DATASET.split('_')[0] + f"_{cfg.MODEL.ARCH}.json"
        if os.path.exists(self.path_source_distri):
            with open(self.path_source_distri) as file:
                self.source_distri = json.load(file)
        else:
            _, self.source_dataloader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                                   root_dir=cfg.DATA_DIR,
                                                   batch_size=128, ckpt_path=cfg.CKPT_PATH,
                                                   workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
            if not os.path.exists(dir_source_distri):
                os.makedirs(dir_source_distri, exist_ok=True)
            logger.info("Pre-compute source distribution statistics...")
            self.precompute_source_distri()
            with open(self.path_source_distri) as file:
                self.source_distri = json.load(file)

    @torch.no_grad()
    def precompute_source_distri(self):

        def avg_batch(data, layer=0):
            result = {'means': [], 'vars': []}
            for name in ['means', 'vars']:
                result[name] = [sum(values) / len(values) for values in zip(*data[name][layer])]
            return result

        model = self.model
        avg_per_batch = {'means': [], 'vars': []}
        for data in tqdm.tqdm(self.source_dataloader):
            x = data[0].cuda()
            for m in model.modules():
                if isinstance(m, MyBatchNorm):
                    m.reset_statistic()
            _ = model(x)
            stat = list(self.stat_outputs.values())

            stats = [item for pair in stat for item in pair]
            # print(len(stats))  # 53 * 2 len if imagenet(resnet50), 31 * 2 if cifar100C, 25 * 2 if cifar10c, 170*2 if res2net
            for i in range(len(stats)):
                stats[i] = stats[i].cpu().tolist()
            stats_one_batch = {'means': [], 'vars': []}
            for i, stat in enumerate(stats):
                if i % 2 == 0:
                    stats_one_batch['means'].append(stat)
                else:
                    stats_one_batch['vars'].append(stat)
            for i in range(len(stats_one_batch['means'])):
                result = avg_batch(stats_one_batch, i)
                if len(avg_per_batch['means']) == len(stats) / 2:
                    avg_per_batch['means'][i] += result['means']
                    avg_per_batch['vars'][i] += result['vars']
                else:  # for first batch
                    avg_per_batch['means'].append(result['means'])
                    avg_per_batch['vars'].append(result['vars'])
        avg_all_batch = {'means': [], 'vars': []}
        for layer in range(len(avg_per_batch['means'])):
            result = avg_batch(stats_one_batch, layer)
            if len(stats_one_batch['means']) == len(stats) / 2:
                avg_all_batch['means'].append(result['means'])
                avg_all_batch['vars'].append(result['vars'])
        assert len(avg_all_batch['vars']) == len(stats) / 2
        assert len(avg_all_batch['means']) == len(stats) / 2

        with open(self.path_source_distri, 'w') as f:
            f.write(json.dumps(
                {
                    "stats": avg_all_batch,
                },
                indent=4,
            ))


    def replace_bn_with_custom(self, custom_bn_class):

        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                custom_bn = custom_bn_class(module, datta_alpha=self.alpha)
                # Determine the parent module and the attribute name
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                    parent_module = modules[parent_name]
                else:
                    attr_name = name
                    parent_module = self.model

                setattr(parent_module, attr_name, custom_bn)

    # define hook
    def get_stat_output(self, name):
        def hook(module, input, output):
            self.stat_outputs[name] = module.stat  # Collect the statistic output
        return hook

    # attach hook to get the statistic during forward pass
    def attach_hooks_to_custom_bn(self):
        for name, module in self.model.named_modules():
            if isinstance(module, MyBatchNorm):
                module.register_forward_hook(self.get_stat_output(name))

    def configure_model(self):
        self.replace_bn_with_custom(MyBatchNorm)
        self.attach_hooks_to_custom_bn()
        self.model.train()
        self.model.requires_grad_(False)
        for m in self.model.modules():  # 25, 31 ,53
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True

        return self.model

    def reset(self):
        # print(type(self.model_states), self.optimizer_state)
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

        # reset the initial statistics of BNs
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """
        Forward and adapt model on batch of data.
        """

        outputs = model(x)
        stat = list(self.stat_outputs.values())

        # confidence threshold
        confidence = torch.softmax(outputs, dim=1)
        outputs_above_threshold = []
        for j in range(confidence.shape[0]):
            if torch.max(confidence[j]) > self.theta:
                outputs_above_threshold.append(outputs[j])

        loss_stat = stat_loss(stat, self.source_distri['stats'])
        if len(outputs_above_threshold) != 0:
            outputs_above_threshold = torch.stack(outputs_above_threshold)
            loss_em = softmax_entropy(outputs_above_threshold).mean(0)
            loss = loss_stat + loss_em
        else:
            loss = loss_stat

        # domain shift detection
        if "continual" in self.setting:  # continual domain shifts
            self.signal, self.loss_list, self.window = self.shift_detector(self.loss_list, self.window,
                                                                                       loss_stat.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # domain shift detection
            if self.signal:
                logging.info('Reset BN layers by domain shift detection.')
                with torch.no_grad():
                    self.reset()
                    _ = model(x)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.stat_outputs = {}
        return outputs

    def shift_detector(self, loss_list, window, loss_stat, length=15, length_win=3):
        length = self.W_l
        length_win = self.W_s
        tau = self.tau
        if len(loss_list) < length:
            loss_list.append(loss_stat)
            if len(window) < length_win:
                window.append(loss_stat)
            else:
                window.pop(0)
                window.append(loss_stat)
            return False, loss_list, window
        else:
            avg_loss = sum(loss_list) / len(loss_list)
            avg_window = sum(window) / len(window)
            if (avg_window - avg_loss) > avg_loss * (tau - 1):
                self.counter += 1
                if self.counter == 1:
                    loss_list = [loss_stat]
                    window = [loss_stat]
                    self.counter = 0
                    return True, loss_list, window

                else:
                    return False, loss_list, window

            else:
                loss_list.pop(0)
                loss_list.append(loss_stat)
                window.pop(0)
                window.append(loss_stat)
                return False, loss_list, window

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs


    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, MyBatchNorm):
                for np, p in m.named_parameters():
                    if p.requires_grad == True:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        print("collect:", names)
        return params, names

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_states, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


l1loss = nn.L1Loss(reduction='mean')
def stat_loss(stat, source_stat):
    for i in range(len(stat)):
        bs = stat[0][0].shape[0]
        if i == 0:
            losses_aff_mean = l1loss(stat[i][0], torch.tensor(source_stat['means'][i]).unsqueeze(0).repeat(bs, 1).cuda())
            losses_aff_var = l1loss(stat[i][1], torch.tensor(source_stat['vars'][i]).unsqueeze(0).repeat(bs, 1).cuda())
        else:
            losses_aff_mean = torch.add(losses_aff_mean, l1loss(stat[i][0], torch.tensor(source_stat['means'][i]).unsqueeze(0).repeat(bs, 1).cuda()))
            losses_aff_var = torch.add(losses_aff_var, l1loss(stat[i][1], torch.tensor(source_stat['vars'][i]).unsqueeze(0).repeat(bs, 1).cuda()))

    return losses_aff_mean + losses_aff_var









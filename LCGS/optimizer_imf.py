import copy
import torch.optim as optim
import torch
import math
from torch.autograd import grad
from utils import *
import numpy as np

class Hyper_SGD(optim.SGD):
    def __init__(self, params, lr, logger, momentum=0, dampening=0, weight_decay=0, nesterov=False,config=None):
        super(Hyper_SGD, self).__init__(params, lr, momentum=momentum, dampening=dampening,
                                         weight_decay=weight_decay, nesterov=nesterov)
        self.K = config.K
        self.steps = config.steps
        self.use_fix_gamma = config.use_fix_gamma
        logger.info("steps:{}, K:{}".format(self.steps, self.K))
        if self.use_fix_gamma:
            self.gamma = config.gamma
            logger.info("use_fix_gamma:{}, gamma:{}".format(self.use_fix_gamma, self.gamma))
        logger.info("use_fix_gamma:{}".format(self.use_fix_gamma))

        # used for gradient clip
        self.grad_clipping = config.clipping
        self.max_norm = 5.0

        # used for counter nan hg
        self.nan_counter = 0

    def power_iteration(self, f_flatten, W_list, init_vec=None, error_threshold=1e-4, momentum=0):
        if init_vec is None:
            vec = torch.rand(f_flatten.size()).to(f_flatten)
        else:
            vec = init_vec

        prev_lambda = 0.0
        for i in range(self.steps):
            prev_vec = vec / (torch.norm(vec) + 1e-6)
            hvp = grad(f_flatten, W_list, grad_outputs=vec, create_graph=True, retain_graph=True)
            hvp_flatten = torch.cat([g_i.contiguous().view(-1) for g_i in hvp])
            new_vec = hvp_flatten - momentum * prev_vec
            lambda_estimate = (vec @ new_vec).item()
            diff = lambda_estimate - prev_lambda
            vec = new_vec.detach() / torch.norm(new_vec)
            if lambda_estimate == 0.0:  # for low-rank
                error = 1.0
            else:
                error = np.abs(diff / lambda_estimate)

            if error < error_threshold:
                return lambda_estimate, vec
            prev_lambda = lambda_estimate

        return lambda_estimate, vec

    def approxInverseHVP(self, v, f, W_list):
        # Neumann approximation of inverse-Hessian-vector product
        """

        :param v: the gradient of evaluation loss with respect to w
        :param f: the gradient of training loss with respect to w
        :param lr: learning rate of inner objective
        :param W_list: a list of model parameters w
        :return: the approximation of inverse-Hessian-vector product p
        """
        p = copy.deepcopy(v)

        if self.use_fix_gamma == False:
            f_flatten = torch.cat([f_i.view(-1) for f_i in f])
            lambda_max, _ = self.power_iteration(f_flatten, W_list)

            if lambda_max <= 0:
                self.gamma = 20
            else:
                self.gamma = 1.0 / lambda_max
            # print("gamma:", self.gamma, " lambda_max:", lambda_max)

        for j in range(self.K):
            hvp = grad(f, W_list, grad_outputs=v, create_graph=True, retain_graph=True)
            # hvp = []
            # for f_i_flatten, v_i in zip(f_flatten, v):
            #     v_i_flatten = v_i.view(-1)
            #     hvp_i_flatten = v_i_flatten@f_i_flatten*f_i_flatten
            #     hvp.append(hvp_i_flatten.view(v_i.shape[0], v_i.shape[1])) # v * f * f^T

            # print("j:{}, hvp_norm:{}".format(j, self.get_norm(hvp)))
            for v_i, hvp_i, p_i in zip(v, hvp, p):
                v_i -= self.gamma * hvp_i.detach()
                p_i += v_i
            del hvp

        for p_i in p:
            p_i *= self.gamma
        return p

    def compute_hypergradient(self, generator, model, features, ys, train_mask, val_mask, l2_reg):
        """

        :param sample_func:
        :param hyper_probs:
        :param model:
        :param features:
        :param deque: a deque, contains ordereddict('W', 'm_t', 'v_t')
        :param ys:
        :param train_mask:
        :param val_mask:
        :param io_steps:
        :return:
        """
        theta_list = self.get_parameters()

        # model.eval()
        adj_t, normalized = generator(mask=True)
        out, _, _ = model(features, adj_t, normalized=normalized)
        val_loss = masked_loss(out, ys, val_mask)

        v1 = grad(val_loss, model.W, create_graph=True, retain_graph=True)
        v1 = [v1_i.detach() for v1_i in v1]

        model.train()
        # adj_s, normalized = generator(mask=True)
        adj_s = adj_t
        out, _, _ = model(features, adj_s, normalized=normalized)
        train_loss_s = masked_loss(out, ys, train_mask)
        train_loss_s += l2_reg * model.l2_loss()
        f = grad(train_loss_s, model.W, create_graph=True, retain_graph=True)

        v2 = self.approxInverseHVP(v1, f, model.W)
        del v1
        # v2 = self.clip_grad(v2)

        v3 = grad(f, theta_list, grad_outputs=v2, create_graph=True, retain_graph=True)
        del f

        direct_grad = grad(val_loss, theta_list)
        hg_grad = [(direct_i - v3_i).detach() for direct_i, v3_i in zip(direct_grad, v3)]
        return hg_grad, self.get_norm(hg_grad)

    def Hyper_step(self, generator, model, features, ys, train_mask, val_mask, l2_reg, logger):
        hg_grad, hg_norm = self.compute_hypergradient(generator, model, features, ys, train_mask,
                                                      val_mask, l2_reg)
        self.assign_hypergradient(hg_grad, logger)
        super(Hyper_SGD, self).step()

    def assign_hypergradient(self, gradients, logger):
        # process nan data in hypergradient
        gradients = self.clip_grad(gradients)
        for i, grad_i in enumerate(gradients):
            if torch.isnan(grad_i).float().sum() > 0:
                grad_i = torch.zeros_like(grad_i)
                gradients[i] = grad_i
                self.nan_counter += 1
                logger.info("Trigger hypergradient nan warning! Nan counter {}".format(self.nan_counter))

        params = self.get_parameters()
        for grad_i, param in zip(gradients, params):
            param.grad = grad_i

    def get_norm(self, list_tensor):
        total_norm = 0
        for tensor in list_tensor:
            tensor_norm = tensor.norm(2).item()
            total_norm += tensor_norm ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def clip_grad(self, gradients, max_norm=None):
        # gradients: list([tensor])
        total_norm = 0
        for gradient in gradients:
            grad_norm = gradient.norm(2).item()
            total_norm += grad_norm ** 2

        total_norm = total_norm ** (1. / 2)
        if max_norm is None:
            max_norm = self.max_norm
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1 and self.grad_clipping:
            for gradient in gradients:
                gradient.mul_(clip_coef)

        return gradients

    def get_parameters(self):
        params = []
        for param_group in self.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

class Adam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(Adam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def flatten_state(self, net):
        return (torch.cat([self.state[v]['exp_avg'].clone().view(-1) for v in net.W]),
                torch.cat([self.state[v]['exp_avg_sq'].clone().view(-1) for v in net.W]),
                [self.state[v]['step'] for v in net.W])

    def show_parameters(self):
        for param_group in self.param_groups:
            return param_group['params'][-1]

    def get_lr(self):
        for param_group in self.param_groups:
            return param_group['lr']
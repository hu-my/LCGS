import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil

_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def lower_triangular_mask(shape, as_array=False):
    a = np.zeros(shape)
    for i in range(1, shape[0]):
        for j in range(0, i):
            a[i, j] = 1.
    # make sure that torch.from_numpy(a) is with no_grad
    return torch.from_numpy(a).float() if not as_array else a

def clear_grad(model):
    for p in model.W:
        if p.grad is not None:
            p.grad = None
            # p.grad.detach()
            # p.grad.zero_()

def empirical_mean_model(generator, net, features, ys, mask, device, eval=False):
    if eval:
        net.eval()
        generator.eval()
    with torch.no_grad():
        adj_hyp, normalized = generator()
        out, _, _ = net(features, adj_hyp, normalized=normalized)
        acc = masked_acc(out, ys, mask)
    return acc

def early_stopping(patience, logger, maxiters=1e10, on_accept=None, on_refuse=None, on_close=None, verbose=True):
    """
    Generator that implements early stopping. Use `send` method to give to update the state of the generator
    (e.g. with last validation accuracy)

    :param patience:
    :param maxiters:
    :param on_accept: function to be executed upon acceptance of the iteration
    :param on_refuse: function to be executed when the iteration is rejected (i.e. the value is lower then best)
    :param on_close: function to be exectued when early stopping activates
    :param verbose:
    :return: a step generator
    """
    val = None
    pat = patience
    t = 0
    while pat and t < maxiters:
        new_val = yield t
        if new_val is not None:
            if val is None or new_val > val:
                val = new_val
                pat = patience
                if on_accept:
                    try:
                        on_accept(t, val)
                    except TypeError:
                        try:
                            on_accept(t)
                        except TypeError:
                            on_accept()
                if verbose: logger.info('ES t={}: Increased val accuracy: {}'.format(t, val))
            else:
                pat -= 1
                if on_refuse: on_refuse(t)
        else:
            t += 1
    yield
    if on_close: on_close(val)
    if verbose: logger.info('ES: ending after {} iterations'.format(t))

def grid_param(io_lr, oo_lr):
    grd = np.array(np.meshgrid(io_lr, oo_lr)).T.reshape(-1, 2)
    return [pair for pair in grd]

def cal_edge_num(generator, n_sample):
    edges = [generator()[0].sum() for _ in range(n_sample)]

    edge = 1.0 * sum(edges) / n_sample
    edge = edge / 2
    return edge

def check_probs(probs):
    assert len(probs.shape) == 2
    num = probs.shape[0]

    for i in range(num):
        for j in range(i+1):
            assert probs[i][j] == 0

def get_value_from_dict(param_dict):
    params = []
    for key, value in param_dict.items():
        params.append(value)
    return params

def drop_edge(adj, ratio=0.4):
    nnz = adj.nonzero()
    nnz_num = nnz.shape[0]
    del_num = int(nnz_num * ratio)
    _, ind = torch.sort(torch.rand(nnz_num).to(adj), descending=True)
    del_ind = nnz[ind[:del_num]]
    # adj[del_ind[:, 0], del_ind[:, 1]] = 0

    mask = (adj > 0).float().detach()
    mask[del_ind[:, 0], del_ind[:, 1]] = 0
    return adj * mask

def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss

def masked_acc(out, label, mask):
    #[node, feature]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc

def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape:
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    #[2, 4926] -> [49216, 2] -> [remained node, 2] ->[2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i,v,x.shape).to(x.device)
    out = out * (1./(1-rate))
    return out

def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
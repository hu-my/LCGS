from data.data_config import EdgeDelConfigData
import time
from logger import setup_logger
import datetime
from generator import Generator
from model import GCN
from utils import *
from optimizer_imf import Hyper_SGD, Adam
from config import CoraConfig, CiteseerConfig, PubmedConfig

def lcgs(io_lr, oo_lr, data, data_config, logger):
    device = torch.device('cuda')
    if data == 'pubmed' and os.path.exists('pubmed_data.npz'):
        npfile = np.load('pubmed_data.npz')
        adj_mods = npfile['adj']
        features = npfile['features']
        ys = npfile['label']
        train_mask, val_mask, es_mask, test_mask = npfile['train_mask'], npfile['val_mask'], npfile['es_mask'], npfile[
            'test_mask']
    else:
        _, adj_mods, features, ys, train_mask, val_mask, es_mask, test_mask = data_config.load()
        if data == 'pubmed':
            np.savez('pubmed_data', adj=adj_mods, features=features, label=ys, train_mask=train_mask, val_mask=val_mask,
                     es_mask=es_mask, test_mask=test_mask)
    print(adj_mods.shape)
    print(features.shape)

    adj_mods_tensor = torch.from_numpy(adj_mods).float().to(device)  # used in calculate hyperprobs
    num_classes = ys.shape[1]
    features_tensor = torch.from_numpy(features).float().to(device)
    ys_tensor = torch.from_numpy(ys).long()
    ys_tensor = ys_tensor.argmax(dim=1).to(device)
    train_mask_tensor = torch.from_numpy(train_mask).to(device)
    val_mask_tensor = torch.from_numpy(val_mask).to(device)  # used for outer objective optimization
    es_mask_tensor = torch.from_numpy(es_mask).to(device)  # used for early stop
    test_mask_tensor = torch.from_numpy(test_mask).to(device)

    logger.info("*" * 10 + " Configuration " + "*" * 10)
    logger.info("io_lr = " + "{:.4f}".format(io_lr) + ", oo_lr = " + "{:.4f}".format(oo_lr))
    logger.info("*" * 35)

    logger.info("num_classes:{}".format(num_classes))
    model = GCN(features_tensor.shape, num_classes, hidden_dim=config.hidden_dim, num_layer=config.num_layer,
                dropout=0.5, device=device)
    generator = Generator(config, adj_mods_tensor)
    generator.to(device)
    del adj_mods_tensor

    inner_optimizer_conf = {'lr': io_lr, 'betas': (0.9, 0.999), 'eps': 1e-5}
    outer_optimizer = Hyper_SGD(generator.parameters(), lr=oo_lr, logger=logger, config=config)
    # outer_scheduler = LambdaLR(outer_optimizer, lr_lambda=lambda epoch: 1. / (1 + 0.001 * epoch))
    logger.info("io_step:{}".format(config.io_steps))

    # reinitialize model parameters and construct optimizer
    model.init_weight()
    # model.init_weight()
    inner_optimizer = Adam(model.W, lr=inner_optimizer_conf['lr'], betas=inner_optimizer_conf['betas'],
                            eps=inner_optimizer_conf['eps'])

    # inner problem stop condition
    best_es_loss = 100
    best_es_acc, best_test_acc = 0, 0
    t = 0
    while t < config.io_params[2]:
        model.train()
        generator.train()
        adj_t, normalized = generator(mask=True)
        out, _, _ = model(features_tensor, adj_t, normalized=normalized)
        train_loss = masked_loss(out, ys_tensor, train_mask_tensor)
        train_loss += config.l2_reg * model.l2_loss()

        inner_optimizer.zero_grad()
        train_loss.backward()
        inner_optimizer.step()
        inner_optimizer.zero_grad()
        t = t + 1

        with torch.no_grad():
            val_loss = masked_loss(out, ys_tensor, val_mask_tensor)
            es_loss = masked_loss(out, ys_tensor, es_mask_tensor)
            train_acc = masked_acc(out, ys_tensor, train_mask_tensor)
            es_acc = masked_acc(out, ys_tensor, es_mask_tensor)

            if t % 10 == 0:
                print("epoch:", t, " train_loss:{:.4f}".format(train_loss.item()),
                      " val_loss:{:.4f}".format(val_loss),
                      " train_acc:{:.4f}".format(train_acc),
                      " es_acc:{:.4f}".format(es_acc),
                      " es_loss:{:.4f}".format(es_loss),
                      " max_memory:{:.4f}".format(torch.cuda.max_memory_allocated() / 1024. / 1024.))

        if best_es_loss > es_loss:
            best_es_loss = es_loss
            test_acc = empirical_mean_model(generator, model, features_tensor, ys_tensor,
                                            test_mask_tensor, device, eval=True)
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        if es_acc > best_es_acc:
            best_es_acc = es_acc
            test_acc = empirical_mean_model(generator, model, features_tensor, ys_tensor,
                                            test_mask_tensor, device, eval=True)
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        del train_loss
        del out
        del adj_t

        if config.io_steps == 0 or t % config.io_steps == 0:
            if t < config.io_steps:
                continue

            # calculcate hypergradient and  update hyper_probs
            outer_optimizer.Hyper_step(generator, model, features_tensor, ys_tensor, train_mask_tensor,
                                        val_mask_tensor, config.l2_reg, logger)
            # outer_scheduler.step()
            clear_grad(model)

    es_final_value = empirical_mean_model(generator, model, features_tensor, ys_tensor,
                                          es_mask_tensor, device, eval=True)
    test_acc = empirical_mean_model(generator, model, features_tensor, ys_tensor,
                                        test_mask_tensor, device, eval=True)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    return es_final_value, best_test_acc

def main(data, seed, name, config):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    print("seed:{}".format(seed))

    svname = name
    if svname is None:
        svname = '{}_{}'.format(data, seed)
    save_path = os.path.join('./save', svname)
    ensure_path(save_path)
    set_log_path(save_path)
    print("save_path:", save_path)
    logger = setup_logger(name, save_path)

    if data == 'cora' or data == 'citeseer' or data == 'pubmed':
        data_config = EdgeDelConfigData(prob_del=0, seed=seed, enforce_connected=False,
                                        dataset_name=data)
    else:
        raise AttributeError('Dataset {} is not available'.format(data))

    best_valid_acc = 0
    best_test_acc = 0
    best_config = {"io_lr":0, "oo_lr":0}

    grd = grid_param(config.io_lr, config.oo_lr)

    start_training_time = time.time()
    for io_lr, oo_lr in grd:
        valid_acc, test_acc = lcgs(io_lr, oo_lr, data, data_config, logger)
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            best_config["io_lr"] = io_lr
            best_config["oo_lr"] = oo_lr

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {}".format(
            total_time_str
        )
    )

    logger.info("best_acc:{}".format(best_test_acc))
    logger.info("best_config:{}".format(best_config))
    logger.info("*"*10)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='method')
    parser.add_argument('--dataset', default='cora', type=str,
                        help='The evaluation dataset: cora, citeseer, pubmed. Default: cora')
    parser.add_argument('--seed', default=1, type=int,
                        help='The random seed. Default: 1')
    parser.add_argument('--gpu', default='0', type=str,
                        help='The gpu device number (must be a single number)')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    if args.dataset == 'cora':
        config = CoraConfig()
    elif args.dataset == 'citeseer':
        config = CiteseerConfig()
    elif args.dataset == 'pubmed':
        config = PubmedConfig()
    else:
        AssertionError("Not implement!")
    set_gpu(args.gpu)

    _data, _seed, _name = args.dataset, args.seed, args.name
    main(_data, _seed, _name, config)





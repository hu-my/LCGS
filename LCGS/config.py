from argparse import ArgumentParser

class Config:
    def __init__(self):
        self.num_layer = 2
        self.hidden_dim = 16
        self.l2_reg = 5.e-4  # l2 regularization coefficient (as in Kipf, 2017 paper)

        self.io_steps = 10  # number of steps of the inner optimization dynamics before an update
        self.steps = 10  # number of steps of power iteration for estimating maximum eigenvalue of Hessian
        self.use_fix_gamma = False
        if self.use_fix_gamma: # give a fixed value for gamma
            self.gamma = 5

class CoraConfig(Config):
    def __init__(self, dropout=0.2, self_loop=False, diag_coeff=0.0, k=1):
        # for dual-normalization
        self.self_loop = self_loop
        self.diag_coeff = diag_coeff
        self.epsilon = 1e-4

        # for improved Neumann-IFT
        self.K = k
        self.clipping = True

        self.dropout = dropout  # for generator training
        self.io_params = (0.001, 3, 200)  # minimum decrease coeff, patience, maxiters

        # meshgrid
        # self.io_lr = (2.e-2, 1.e-1, 0.05) # learning rate for the inner optimizer
        # self.oo_lr = (0.05, .1, 1.) # learning rate of the outer optimizer
        self.io_lr = (2.e-2,)
        self.oo_lr = (0.05,)

        super(CoraConfig, self).__init__()


class CiteseerConfig(Config):
    def __init__(self, dropout=0.0, self_loop=True, diag_coeff=1.0, k=10):
        # for dual-normalization
        self.self_loop = self_loop
        self.diag_coeff = diag_coeff
        self.epsilon = 0

        # for improved Neumann-IFT
        self.K = k
        self.clipping = True

        self.dropout = dropout  # for generator training
        self.io_params = (0.001, 3, 200)  # minimum decrease coeff, patience, maxiters

        # meshgrid
        # self.io_lr = (2.e-2, 1.e-1, 0.05) # learning rate for the inner optimizer
        # self.oo_lr = (.05, .1, 1.) # learning rate of the outer optimizer
        self.io_lr = (2.e-2,)
        self.oo_lr = (.05,)

        super(CiteseerConfig, self).__init__()

class PubmedConfig(Config):
    def __init__(self, dropout=0.2, self_loop=True, diag_coeff=0.0, k=30):
        # for dual-normalization
        self.self_loop = self_loop
        self.diag_coeff = diag_coeff
        self.epsilon = 1e-4

        # for improved Neumann-IFT
        self.K = k
        self.clipping = True

        self.dropout = dropout  # for generator training
        self.io_params = (0.001, 20, 400)  # minimum decrease coeff, patience, maxiters

        # meshgrid
        # self.io_lr = (2.e-2, 1.e-1, 0.05) # learning rate for the inner optimizer
        # self.oo_lr = (.001, .01, .1) # learning rate of the outer optimizer
        self.io_lr = (2.e-2,)
        self.oo_lr = (.001,)

        super(PubmedConfig, self).__init__()
from .utils import load_data, upper_triangular_mask
import numpy as np
import scipy.sparse as sp
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))
    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items())) + ']\n'

    @classmethod
    def default_instance(cls):
        return cls()

class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        self.f1 = 'load_data_del_edges'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        # res: (adj, del_adj, features, y_train) + other_splittables
        if self.f2:
            res = eval(self.f2)(res, **self.kwargs_f2, seed=self.seed)
        # res: adj, adj_mods, features, ys, train_mask, mask_val, mask_es, test_mask
        return res

class EdgeDelConfigData(ConfigData):
    def __init__(self, **kwargs):
        self.prob_del = kwargs['prob_del']
        self.enforce_connected = kwargs['enforce_connected']
        super().__init__(**kwargs)
        self.kwargs_f1['prob_del'] = self.prob_del
        if not self.enforce_connected:
            self.kwargs_f1['enforce_connected'] = self.enforce_connected
        del self.prob_del
        del self.enforce_connected


class UCI(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        self.n_es = None
        self.scale = None
        super().__init__(**kwargs)

    def load(self):
        if self.dataset_name == 'iris':
            data = datasets.load_iris()
        elif self.dataset_name == 'wine':
            data = datasets.load_wine()
        elif self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
        elif self.dataset_name == 'digits':
            data = datasets.load_digits()
        elif self.dataset_name == 'fma':
            import os
            data = np.load('%s/fma/fma.npz' % os.getcwd())
        elif self.dataset_name == '20news10':
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            categories = ['alt.atheism',
                          'comp.sys.ibm.pc.hardware',
                          'misc.forsale',
                          'rec.autos',
                          'rec.sport.hockey',
                          'sci.crypt',
                          'sci.electronics',
                          'sci.med',
                          'sci.space',
                          'talk.politics.guns']
            data = fetch_20newsgroups(subset='all', categories=categories)
            vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
            X_counts = vectorizer.fit_transform(data.data).toarray()
            transformer = TfidfTransformer(smooth_idf=False)
            features = transformer.fit_transform(X_counts).todense()
        else:
            raise AttributeError('dataset not available')

        if self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if self.scale:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
        ys = LabelBinarizer().fit_transform(y)
        if ys.shape[1] == 1:
            ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]
        from sklearn.model_selection import train_test_split
        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val + self.n_es,
                                                        test_size=n - self.n_train - self.n_val - self.n_es,
                                                        stratify=y)
        train, es, y_train, y_es = train_test_split(train, y_train, random_state=self.seed,
                                                    train_size=self.n_train + self.n_val, test_size=self.n_es,
                                                    stratify=y_train)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                      train_size=self.n_train, test_size=self.n_val,
                                                      stratify=y_train)

        train_mask = np.zeros([n, ], dtype=bool)
        train_mask[train] = True
        val_mask = np.zeros([n, ], dtype=bool)
        val_mask[val] = True
        es_mask = np.zeros([n, ], dtype=bool)
        es_mask[es] = True
        test_mask = np.zeros([n, ], dtype=bool)
        test_mask[test] = True

        return np.zeros([n, n]), np.zeros([n, n]), features, ys, train_mask, val_mask, es_mask, test_mask

def graph_delete_connections(prob_del, seed, adj, features, y_train,
                             *other_splittables, to_dense=False,
                             enforce_connected=False):
    rnd = np.random.RandomState(seed)

    features = preprocess_features(features)

    if to_dense:
        features = features.toarray()
        adj = adj.toarray()
    print("# INIT EGDES:", np.sum(adj))
    del_adj = np.array(adj, dtype=np.float32)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * upper_triangular_mask(
        adj.shape, as_array=True)
    smpl += smpl.transpose()

    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)
    return (adj, del_adj, features, y_train) + other_splittables

def load_data_del_edges(prob_del=0.4, seed=0, to_dense=True, enforce_connected=True,
                        dataset_name='cora'):
    res = graph_delete_connections(prob_del, seed, *load_data(dataset_name), to_dense=to_dense,
                                   enforce_connected=enforce_connected)
    return res


def reorganize_data_for_es(loaded_data, seed=0, es_n_data_prop=0.5):
    adj, adj_mods, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = loaded_data
    ys = y_train + y_val + y_test
    features = preprocess_features(features)
    msk1, msk2 = divide_mask(es_n_data_prop, np.sum(val_mask), seed=seed)
    mask_val = np.array(val_mask)
    mask_es = np.array(val_mask)
    mask_val[mask_val] = msk2
    mask_es[mask_es] = msk1

    return adj, adj_mods, features, ys, train_mask, mask_val, mask_es, test_mask


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def divide_mask(n1, n_tot, seed=0):
    rnd = np.random.RandomState(seed)
    p = n1 / n_tot if isinstance(n1, int) else n1
    chs = rnd.choice([True, False], size=n_tot, p=[p, 1. - p])
    return chs, ~chs
"""
Source:
    https://gist.github.com/davidbau/00a9b6763a260be8274f6ba22df9a145
"""
import os
import math
import struct
import typing as T
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader


null_numpy_value = np.array(
    struct.unpack('>d', struct.pack('>Q', 0xfff8000000000002))[0],
    dtype=np.float64
)


def is_null_numpy_value(v) -> bool:
    return (
        isinstance(v, np.ndarray) and np.ndim(v) == 0
        and v.dtype == np.float64 and np.isnan(v)
        and 0xfff8000000000002 == struct.unpack('>Q', struct.pack('>d', v))[0]
    )


def box_numpy_null(d):
    try:
        return {k: box_numpy_null(v) for k, v in d.items()}
    except:
        return null_numpy_value if d is None else d


def unbox_numpy_null(d):
    try:
        return {k: unbox_numpy_null(v) for k, v in d.items()}
    except:
        return None if is_null_numpy_value(d) else d


def resolve_state_dict(s):
    """Resolves a state, which can be a filename or a dict-like object.
    """
    if isinstance(s, str):
        return unbox_numpy_null(np.load(s))
    return s


def save_cached_state(cachefile, obj, args):
    if cachefile is None:
        return
    dat = obj.state_dict()
    for a, v in args.items():
        if a in dat:
            assert (dat[a] == v)
        dat[a] = v
    if isinstance(cachefile, dict):
        cachefile.clear()
        cachefile.update(dat)
    else:
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        np.savez(cachefile, **box_numpy_null(dat))


global_load_cache_enabled = True
def load_cached_state(
    cachefile: T.Union[Path, str],
    args: T.Optional[dict] = None,
    quiet: T.Optional[bool] = False,
    throw: T.Optional[bool] = False
):
    """Resolves a state, which can be a filename or a dict-like object.
    """
    if args is None:
        args = {}
    if not global_load_cache_enabled or cachefile is None:
        return None
    try:
        if isinstance(cachefile, dict):
            dat = cachefile
            cachefile = 'state' # for printed messages
        else:
            dat = unbox_numpy_null(np.load(cachefile))
        for a, v in args.items():
            if a not in dat or dat[a] != v:
                print(
                    '%s %s changed from %s to %s' % (cachefile, a, dat[a], v)
                )
                return None
    except (FileNotFoundError, ValueError) as e:
        if throw:
            raise e
        return None
    else:
        if not quiet:
            print('Loading cached %s' % cachefile)
        return dat


class Stat(object):
    """Abstract base class for a running pytorch statistic.
    """
    def __init__(self, state):
        """By convention, all Stat subclasses can be initialized by passing
        state=; and then they will initialize by calling load_state_dict.
        """
        self.load_state_dict(resolve_state_dict(state))

    def add(self, x, *args, **kwargs):
        """Observes a batch of samples to be incorporated into the statistic.
        Dimension 0 should be the batch dimension, and dimension 1 should
        be the feature dimension of the pytorch tensor x.
        """
        pass

    def load_state_dict(self, d):
        """Loads this Stat from a dictionary of numpy arrays as saved
        by state_dict.
        """
        pass

    def state_dict(self):
        """Saves this Stat as a dictionary of numpy arrays that can be
        stored in an npz or reloaded later using load_state_dict.
        """
        return {}

    def save(self, filename):
        """Saves this stat as an npz file containing the state_dict.
        """
        save_cached_state(filename, self, {})

    def load(self, filename):
        """
        Loads this stat from an npz file containing a saved state_dict.
        """
        self.load_state_dict(
            load_cached_state(filename, {}, quiet=True, throw=True)
        )

    def to_(self, device):
        """Moves this Stat to the given device.
        """
        pass

    def cpu_(self):
        """Moves this Stat to the cpu device.
        """
        self.to_('cpu')

    def cuda_(self):
        """Moves this Stat to the default cuda device.
        """
        self.to_('cuda')

    def _normalize_add_shape(self, x, attr='data_shape'):
        """Flattens input data to 2d.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if len(x.shape) < 1:
            x = x.view(-1)
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            data_shape = x.shape[1:]
            setattr(self, attr, data_shape)
        else:
            assert x.shape[1:] == data_shape

        return x.view(x.shape[0], int(np.prod(data_shape)))

    def _restore_result_shape(self, x, attr='data_shape'):
        """Restores output data to input data shape.
        """
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            return x

        return x.view(data_shape * len(x.shape))


class Mean(Stat):
    """Running mean
    """
    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)

        self.count = 0
        self.batchcount = 0
        self._mean = None
        self.data_shape = None

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        centered = a - batch_mean
        self.batchcount += 1
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            return
        # Update a batch using Chan-style update for numerical stability.
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)

    def size(self):
        return self.count

    def mean(self):
        return self._restore_result_shape(self._mean)

    def to_(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)

    def load_state_dict(self, state):
        self.count = state['count']
        self.batchcount = state['batchcount']
        self._mean = torch.from_numpy(state['mean'])
        self.data_shape = None if state['data_shape'] is None else tuple(state['data_shape'])

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' + self.__class__.__name__ + '()',
            count=self.count,
            data_shape=self.data_shape and tuple(self.data_shape),
            batchcount=self.batchcount,
            mean=self._mean.cpu().numpy()
        )


class Variance(Stat):
    """Running computation of mean and variance. Use this when you just need
    basic stats without covariance.
    """
    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)

        self.count = 0
        self.batchcount = 0
        self._mean = None
        self.v_cmom2 = None
        self.data_shape = None

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        centered = a - batch_mean
        self.batchcount += 1
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            self.v_cmom2 = centered.pow(2).sum(0)
            return
        # Update a batch using Chan-style update for numerical stability.
        oldcount = self.count
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)
        # Update the variance using the batch deviation
        self.v_cmom2.add_(centered.pow(2).sum(0))
        self.v_cmom2.add_(delta.pow_(2).mul_(new_frac * oldcount))

    def size(self):
        return self.count

    def mean(self):
        return self._restore_result_shape(self._mean)

    def var(self, unbiased=True):
        return self._restore_result_shape(
            self.v_cmom2 / (self.count - (1 if unbiased else 0))
        )

    def std(self, unbiased=True):
        return self.var(unbiased=unbiased).sqrt()

    def to_(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)
        if self.v_cmom2 is not None:
            self.v_cmom2 = self.v_cmom2.to(device)

    def load_state_dict(self, state):
        self.count = state['count']
        self.batchcount = state['batchcount']
        self._mean = torch.from_numpy(state['mean'])
        self.v_cmom2 = torch.from_numpy(state['cmom2'])
        self.data_shape = None if state['data_shape'] is None else tuple(state['data_shape'])

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' + self.__class__.__name__ + '()',
            count=self.count,
            data_shape=self.data_shape and tuple(self.data_shape),
            batchcount=self.batchcount,
            mean=self._mean.cpu().numpy(),
            cmom2=self.v_cmom2.cpu().numpy()
        )


class Quantile(Stat):
    """Streaming randomized quantile computation for torch.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates be read out using quantile(q).
    Implemented as a sorted sample that retains at least r samples
    (by default r = 3072); the number of retained samples will grow to
    a finite ceiling as the data is accumulated.  Accuracy scales according
    to r: the default is to set resolution to be accurate to better than about
    0.1%, while limiting storage to about 50,000 samples.
    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """
    def __init__(self, r: int = 3072, buffersize: int = None, seed=None, state=None):
        if state is not None:
            return super().__init__(state)

        self.depth = None
        self.dtype = None
        self.device = None
        resolution = r * 2  # sample array is at least half full before discard
        self.resolution = resolution
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = None
        self.firstfree = [0]
        self.randbits = torch.ByteTensor(resolution)
        self.currentbit = len(self.randbits) - 1
        self.extremes = None
        self.count = 0
        self.batchcount = 0

    def size(self):
        return self.count

    def _lazy_init(self, incoming):
        self.depth = incoming.shape[1]
        self.dtype = incoming.dtype
        self.device = incoming.device
        self.data = [
            torch.zeros(
                self.depth,
                self.resolution,
                dtype=self.dtype,
                device=self.device
            )
        ]
        self.extremes = torch.zeros(
            self.depth, 2, dtype=self.dtype, device=self.device
        )
        self.extremes[:, 0] = float('inf')
        self.extremes[:, -1] = -float('inf')

    def to_(self, device):
        """Switches internal storage to specified device.
        """
        if device != self.device:
            old_data = self.data
            old_extremes = self.extremes
            self.data = [d.to(device) for d in self.data]
            self.extremes = self.extremes.to(device)
            self.device = self.extremes.device
            del old_data
            del old_extremes

    def add(self, incoming):
        if self.depth is None:
            self._lazy_init(incoming)
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth, (incoming.shape[1], self.depth)
        self.count += incoming.shape[0]
        self.batchcount += 1
        # Convert to a flat torch array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = int(math.ceil(self.buffersize / self.samplerate))
        for index in range(0, len(incoming), chunksize):
            batch = incoming[index:index + chunksize]
            sample = sample_portion(batch, self.samplerate)
            if len(sample):
                self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        # First time sampling - the data source is very large.
                        self._scan_extremes(incoming)
                    incoming = sample_portion(incoming, self.samplerate)
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:, ff:ff + copycount] = torch.t(
                incoming[index:index + copycount, :]
            )
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
            -(-self.data[index - 1].shape[1] // 2) if index else 1
        ):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:, 0 : self.firstfree[index]]
            data = data.sort()[0]
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:, 0], data[:, -1])
            offset = self._randbit()
            position = self.firstfree[index + 1]
            subset = data[:, offset::2]
            self.data[index + 1][:, position : position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1

        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
            torch.min(incoming, dim=0)[0], torch.max(incoming, dim=0)[0]
        )

    def _update_extremes(self, minr, maxr):
        self.extremes[:, 0] = torch.min(
            torch.stack([self.extremes[:, 0], minr]), dim=0
        )[0]
        self.extremes[:, -1] = torch.max(
            torch.stack([self.extremes[:, -1], maxr]), dim=0
        )[0]

    def _randbit(self):
        self.currentbit += 1
        if self.currentbit >= len(self.randbits):
            self.randbits.random_(to=2)
            self.currentbit = 0
        return self.randbits[self.currentbit]

    def state_dict(self):
        state = dict(
            constructor=self.__module__ + '.' + self.__class__.__name__ + '()',
            resolution=self.resolution,
            depth=self.depth,
            buffersize=self.buffersize,
            samplerate=self.samplerate,
            sizes=np.array([d.shape[1] for d in self.data]),
            extremes=self.extremes.cpu().detach().numpy(),
            size=self.count,
            batchcount=self.batchcount,
        )
        for i, (d, f) in enumerate(zip(self.data, self.firstfree)):
            state[f'data.{i}'] = d.cpu().detach().numpy()[:, :f].T

        return state

    def load_state_dict(self, state):
        self.resolution = int(state['resolution'])
        self.randbits = torch.ByteTensor(self.resolution)
        self.currentbit = len(self.randbits) - 1
        self.depth = int(state['depth'])
        self.buffersize = int(state['buffersize'])
        self.samplerate = float(state['samplerate'])
        firstfree = []
        buffers = []
        for i, s in enumerate(state['sizes']):
            d = state[f'data.{i}']
            firstfree.append(d.shape[0])
            buf = np.zeros((d.shape[1], s), dtype=d.dtype)
            buf[:, : d.shape[0]] = d.T
            buffers.append(torch.from_numpy(buf))
        self.firstfree = firstfree
        self.data = buffers
        self.extremes = torch.from_numpy((state['extremes']))
        self.count = int(state['size'])
        self.batchcount = int(state.get('batchcount', 0))
        self.dtype = self.extremes.dtype
        self.device = self.extremes.device

    def min(self):
        return self.minmax()[0]

    def max(self):
        return self.minmax()[-1]

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, : self.firstfree[0]].t())
        return self.extremes.clone()

    def median(self):
        return self.quantiles(0.5)

    def mean(self):
        return self.integrate(lambda x: x) / self.count

    def var(self, unbiased=True):
        mean = self.mean()[:, None]
        return self.integrate(
            lambda x: (x - mean).pow(2)
        ) / (self.count - (1 if unbiased else 0))

    def std(self, unbiased=True):
        return self.var(unbiased=unbiased).sqrt()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(
                0, torch.zeros(self.depth, cap, dtype=self.dtype, device=self.device)
            )
            self.firstfree.insert(0, 0)
        else:
            # Unless we're so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index - 1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level's buffer size (rounding up)
            if self.data[index - 1].shape[1] - (amount + position) >= (
                -(-self.data[index - 2].shape[1] // 2) if (index - 1) else 1
            ):
                self.data[index - 1][:, position : position + amount] = self.data[
                    index
                ][:, :amount]
                self.firstfree[index - 1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:, :amount]
                data = data.sort()[0]
                if index == 1:
                    self._update_extremes(data[:, 0], data[:, -1])
                offset = self._randbit()
                scrunched = data[:, offset::2]
                self.data[index][:, : scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = int(math.ceil(self.resolution * (0.67 ** len(self.data))))
        if cap < 2:
            return 0
        # Round up to the nearest multiple of 8 for better GPU alignment.
        cap = -8 * (-cap // 8)
        return max(self.buffersize, cap)

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, : self.firstfree[0]].t())
        size = sum(self.firstfree)
        weights = torch.FloatTensor(size)  # Floating point
        summary = torch.zeros(self.depth, size, dtype=self.dtype, device=self.device)
        index = 0
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:, index : index + ff] = self.data[level][:, :ff]
            weights[index : index + ff] = 2.0 ** level
            index += ff
        assert index == summary.shape[1]
        if sort:
            summary, order = torch.sort(summary, dim=-1)
            weights = weights[order.view(-1).cpu()].view(order.shape)
            summary = torch.cat(
                [self.extremes[:, :1], summary, self.extremes[:, 1:]], dim=-1
            )
            weights = torch.cat(
                [
                    torch.zeros(weights.shape[0], 1),
                    weights,
                    torch.zeros(weights.shape[0], 1),
                ],
                dim=-1,
            )

        return (summary, weights)

    def quantiles(self, quantiles):
        if not hasattr(quantiles, 'cpu'):
            quantiles = torch.tensor(quantiles)
        qshape = quantiles.shape
        if self.count == 0:
            return torch.full((self.depth,) + qshape, torch.nan)
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros(
            self.depth, quantiles.numel(), dtype=self.dtype, device=self.device
        )
        # numpy is needed for interpolation
        nq = quantiles.view(-1).cpu().detach().numpy()
        ncw = cumweights.cpu().detach().numpy()
        nsm = summary.cpu().detach().numpy()
        for d in range(self.depth):
            result[d] = torch.tensor(
                np.interp(nq, ncw[d], nsm[d]), dtype=self.dtype, device=self.device
            )

        return result.view((self.depth,) + qshape)

    def integrate(self, fun):
        result = []
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            result.append(torch.sum(fun(self.data[level][:, :ff]) * (2.0 ** level), dim=-1))
        if len(result) == 0:
            return None

        return torch.stack(result).sum(dim=0) / self.samplerate

    def readout(self, count=1001):
        return self.quantiles(torch.linspace(0.0, 1.0, count))

    def normalize(self, data):
        """Given input data as taken from the training distirbution,
        normalizes every channel to reflect quantile values,
        uniformly distributed, within [0, 1].
        """
        assert self.count > 0
        assert data.shape[0] == self.depth
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros_like(data).float()
        # numpy is needed for interpolation
        ndata = data.cpu().numpy().reshape((data.shape[0], -1))
        ncw = cumweights.cpu().numpy()
        nsm = summary.cpu().numpy()
        for d in range(self.depth):
            normed = torch.tensor(
                np.interp(ndata[d], nsm[d], ncw[d]),
                dtype=torch.float,
                device=data.device,
            ).clamp_(0.0, 1.0)
            if len(data.shape) > 1:
                normed = normed.view(*(data.shape[1:]))
            result[d] = normed

        return result


def tally_stats(
    stats: T.Sequence[T.Union[Mean, Variance, Quantile]],
    loader: DataLoader,
    caches: T.Sequence[T.Union[Path, str]],
    quiet: bool = True
):
    """Tally stats

    To use tally_stats, write code like the following.
        ds = EdgeDataset(
            ppaths.train_path,
            processes=4,
            threads_per_worker=2,
            random_seed=100
        )
        train_ds, val_ds = ds.split_train_val(
            val_frac=0.2,
            spatial_overlap_allowed=False
        )
        data_module = EdgeDataModule(
            train_ds=train_ds,
            batch_size=8,
            num_workers=2,
            shuffle=False
        )
        stat = Mean()
        for batch in tally_stats(
            stat, data_module, cache='data_means.npz'
        ):
           stat.add(batch.x)
        mean = stat.mean()

    The first argument should be the Stat being computed. After the
    loader is exhausted, tally will bring this stat to the cpu and
    cache it (if a cache is specified).
    The dataset can be a torch Dataset or a plain Tensor, or it can
    be a callable that returns one of those.

    Details on caching via the cache= argument:
        If the given filename cannot be loaded, tally will leave the
        statistic object empty and set up a DataLoader object so that
        the loop can be run.  After the last iteration of the loop, the
        completed statistic will be moved to the cpu device and also
        saved in the cache file.
        If the cached statistic can be loaded from the given file, tally
        will not set up the data loader and instead will return a fully
        loaded statistic object (on the cpu device) and an empty list as
        the loader.
        The `with cache_load_enabled(False):` context manager can
        be used to disable loading from the cache.

    If needed, a DataLoader will be created to wrap the dataset:
        Keyword arguments of tally are passed to the DataLoader,
        so batch_size, num_workers, pin_memory, etc. can be specified.

    Subsampling is supported via sample_size= and random_sample=:
        If sample_size=N is specified, rather than loading the whole
        dataset, only the first N items are sampled.  If additionally
        random_sample=S is specified, the pseudorandom seed S will be
        used to select a fixed psedorandom sample of size N to sample.
    """
    updated_stats = []
    for stat, cache in zip(stats, caches):
        assert isinstance(stat, (Mean, Quantile, Variance))

        args = {}
        cached_state = load_cached_state(cache, args, quiet=quiet)
        if cached_state is not None:
            stat.load_state_dict(cached_state)
            def empty_loader():
                return
                yield

            return empty_loader()
        updated_stats.append(stat)

    def wrapped_loader():
        yield from loader
        for stat, cache in zip(updated_stats, caches):
            stat.to_(device='cpu')
            if cache is not None:
                save_cached_state(cache, stat, args)

    return wrapped_loader()


class cache_load_enabled():
    """When used as a context manager, cache_load_enabled(False) will prevent
    tally from loading cached statsitics, forcing them to be recomputed.
    """
    def __init__(self, enabled=True):
        self.prev = False
        self.enabled = enabled

    def __enter__(self):
        global global_load_cache_enabled
        self.prev = global_load_cache_enabled
        global_load_cache_enabled = self.enabled

    def __exit__(self, exc_type, exc_value, traceback):
        global global_load_cache_enabled
        global_load_cache_enabled = self.prev

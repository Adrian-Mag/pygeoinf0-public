"""Running blocks of work in parallel (plan 005, P3 will grow this into the execution layer): the number of worker
processes, a fork pool whose workers inherit the data (nothing large is pickled) and run with one BLAS thread each,
and the grouping of measures into blocks of bounded size.

    workers()                          pg.settings.workers (PYGEOINF0_WORKERS, default 1: serial)
    map_blocks(f, args, tasks)         [(t, f(*args, *t)) for t in tasks], in workers when there are several
    imap_blocks(f, args, tasks)        the same pairs one at a time as they finish (in task order when serial), so a
                                       caller can stop early (the pool is terminated when the generator is closed)
    Pool(f, args), shared(shape)       one fork pool for several rounds (arrays changing between rounds shared)
    groups(A, size)                    consecutive runs of the measures of atoms A with at most size atoms each
    in_worker()                        whether this process is a forked worker (where no OpenMP code may run: a child
                                       of a process that used libgomp's thread pool can hang in it, so finufft, which
                                       bundles libgomp, runs in the parent only; realisations/rkhs/spectral.py)
"""
import numpy as np


def workers():
    """Processes for large energy and evaluation blocks: pg.settings.workers (default: the environment variable
    PYGEOINF0_WORKERS, else 1, serial). Blocks run in forked processes (Linux), so the atoms are shared, not pickled."""
    from .settings import settings
    return settings.workers


_JOB = None              # (function, arguments) of the blocks run by a pool: inherited by the forked workers
_WORKER = False          # set in each forked worker by its initialiser


def in_worker():
    """Whether this process is a forked worker of imap_blocks."""
    return _WORKER


def _one_thread():
    """Worker initialiser: one BLAS thread per worker (inherited BLAS threads would spin against each other)."""
    global _WORKER
    _WORKER = True
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _run_block(t):
    f, args = _JOB
    return t, f(*args, *t)


def map_blocks(f, args, tasks):
    """[(t, f(*args, *t)) for t in tasks], in forked worker processes when workers() > 1 and there is more than one."""
    return list(imap_blocks(f, args, tasks))


def imap_blocks(f, args, tasks):
    """The pairs (t, f(*args, *t)) as they finish: in task order when serial, in completion order in the fork pool.
    Closing the generator early (a caller that stops, an exception) terminates the pool."""
    global _JOB
    n = min(workers(), len(tasks))
    if n <= 1:
        for t in tasks:
            yield t, f(*args, *t)
        return
    import multiprocessing as mp
    _JOB = (f, args)
    try:
        with mp.get_context("fork").Pool(n, initializer=_one_thread) as pool:
            yield from pool.imap_unordered(_run_block, tasks)
    finally:
        _JOB = None


def shared(shape):
    """A float array in anonymous shared memory: written by the parent between rounds of a Pool, read by its workers
    (forked earlier, so a fresh array would not reach them)."""
    import mmap
    size = int(np.prod(shape))
    return np.frombuffer(mmap.mmap(-1, max(8 * size, 8)), float, size).reshape(shape)


class Pool:
    """One fork pool for several rounds of blocks over the same arguments: map(tasks) is map_blocks(f, args, tasks),
    but the workers are forked once, at the first round with more than one task, and serve every later round (a fork
    of a large process costs about a second). The arguments are inherited at that fork: arrays that change between
    rounds must be shared (shared(shape)) and written only while no round runs; tasks and results are pickled.
    Serial when workers() <= 1 or in a worker. Close it (or use it as a context manager)."""

    def __init__(self, f, args):
        self.f, self.args, self._pool = f, args, None

    def parallel(self):
        return workers() > 1 and not in_worker()

    def map(self, tasks):
        if not self.parallel() or len(tasks) <= 1:
            return [(t, self.f(*self.args, *t)) for t in tasks]
        if self._pool is None:
            import multiprocessing as mp
            global _JOB
            _JOB = (self.f, self.args)
            try:
                self._pool = mp.get_context("fork").Pool(workers(), initializer=_one_thread)
            finally:
                _JOB = None
        return list(self._pool.imap_unordered(_run_block, tasks))

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def groups(A, size):
    """Consecutive runs of the measures of A with at most `size` atoms each (a larger measure is a run of its own)."""
    counts = np.bincount(A.owner, minlength=A.n)
    out, cur, m = [], [], 0
    for i, c in enumerate(counts):
        if cur and m + c > size:
            out.append(np.array(cur))
            cur, m = [], 0
        cur.append(i)
        m += c
    return out + ([np.array(cur)] if cur else [])

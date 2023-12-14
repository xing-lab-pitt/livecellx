import tqdm
from functools import partial
from multiprocessing import Pool


def wrap_func(func, args):
    if isinstance(args, tuple) or isinstance(args, list):
        return func(*args)
    elif isinstance(args, dict):
        return func(**args)
    else:
        raise TypeError("args must be tuple, list or dict")


def parallelize(func, inputs, cores=None):
    with Pool(processes=cores) as pool:
        outputs = []
        for output in tqdm.tqdm(pool.imap_unordered(partial(wrap_func, func), inputs), total=len(inputs)):
            outputs.append(output)
    return outputs

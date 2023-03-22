import tqdm
from functools import partial
from multiprocessing import Pool


def wrap_func(func, args):
    if isinstance(args, tuple) or isinstance(args, list):
        return func(*args)
    elif isinstance(args, dict):
        return func(**args)


def parallelize(func, inputs, cores=None):
    pool = Pool(processes=cores)
    outputs = []
    for output in tqdm.tqdm(pool.imap_unordered(partial(wrap_func, func), inputs), total=len(inputs)):
        outputs.append(output)
    pool.close()
    pool.join()
    return outputs

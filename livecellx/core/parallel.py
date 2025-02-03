import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count


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


def wrap_func_chunk(func, chunk):
    res = []
    for item in chunk:
        res.append(wrap_func(func, item))
    return res


def parallelize_chunk(func, inputs, cores=None, chunk_size=None):
    # If chunk_size is not specified, use the length of inputs for one big chunk
    # Calculate the number of cores if not provided
    if cores is None:
        cores = cpu_count()

    # If chunk_size is not specified, calculate based on the number of cores
    if chunk_size is None:
        chunk_size = len(inputs) // cores + (len(inputs) % cores > 0)

    # Split inputs into chunks
    chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]

    with Pool(processes=cores) as pool:
        outputs = []
        for output in tqdm.tqdm(pool.imap_unordered(partial(wrap_func_chunk, func), chunks), total=len(chunks)):
            outputs.extend(output)  # Flatten the list of results

    return outputs

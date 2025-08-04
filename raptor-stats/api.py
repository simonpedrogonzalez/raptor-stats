

def zonal_stats(*args, **kwargs):
    """
    should be some rasterstats.zonal_stats-like function.
    
    The primary zonal statistics entry point.

    All arguments are passed directly to ``gen_zonal_stats``.
    See its docstring for details.

    The only difference is that ``zonal_stats`` will
    return a list rather than a generator."""
    progress = kwargs.get("progress")
    if progress:
        if tqdm is None:
            raise ValueError(
                "You specified progress=True, but tqdm is not installed in the environment. You can do pip install rasterstats[progress] to install tqdm!"
            )
        stats = gen_zonal_stats(*args, **kwargs)
        total = sum(1 for _ in stats)
        return [stat for stat in tqdm(stats, total=total)]
    else:
        return list(gen_zonal_stats(*args, **kwargs))
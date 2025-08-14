
from raptorstats.io import (
    open_raster,
    open_vector,
    validate_is_north_up,
    validate_raster_vector_compatibility,
)
from raptorstats.scanline import Scanline
from raptorstats.stats import Stats
from raptorstats.agqt import AggQuadTree


def zonal_stats(
    vectors,
    raster,
    stats=None,
    layer=0,
    band=1,
    nodata=None,
    affine=None,
    prefix=None,
    categorical=False,
    method="scanline",
    **kwargs,
):
    """Zonal statistics using the AggQuadTree or Scanline methods.

    Parameters
    ----------
    vectors :  str | GeoDataFrame | anything GeoPandas can read
    raster  :  str | path-like | rasterio.DatasetReader
    stats   :  list[str]            statistics to compute (min, max, mean, etc.)
    layer   :  int | str            layer name/number for multi-layer vector sources
    band    :  int                  raster band (1-based)
    nodata  :  float | None         overrides raster NODATA value
    affine  :  affine.Affine | None required if `raster` is a NumPy array (not used here)
    prefix  :  str | None           prefix every stat key (useful for merges)
    categorical : bool              whether to compute categorical stats (histogram)
    method  :  str                  zonal statistics method to use ('agqt' or 'scanline')

    Returns
    -------
    list[dict]                      one stats-dict (or Feature) per input geometry

    """

    stats_conf = Stats(stats, categorical=categorical)

    if method == "agqt":
        func = AggQuadTree(**kwargs)
    elif method == "scanline":
        func = Scanline(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'agqt' or 'scanline'.")


    with open_raster(raster, affine=affine, nodata=nodata, band=band) as ds:

        gdf = open_vector(vectors, layer=layer, affine=affine)

        validate_is_north_up(ds.transform)
        validate_raster_vector_compatibility(ds, gdf)

        results = func(ds, gdf, stats=stats_conf)

        results = stats_conf.clean_results(results)

        if prefix:
            results = [
                {f"{prefix}{k}": v for k, v in res.items()} if res is not None else None
                for res in results
            ]

        return results

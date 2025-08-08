import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from shapely import wkt
from shapely.geometry.base import BaseGeometry


def validate_raster_vector_compatibility(
    raster: rio.DatasetReader, vector: gpd.GeoDataFrame
):
    raster_crs = raster.crs
    vector_crs = vector.crs

    if not str(raster_crs) == str(vector_crs):
        warnings.warn(
            "Raster and vector CRS do not match. This may lead to incorrect results.",
            UserWarning,
        )


def validate_is_north_up(transform):
    not_north_up = (
        not np.isclose(transform.b, 0)
        or not np.isclose(transform.d, 0)
        or transform.e > 0
    )
    if not_north_up:
        warnings.warn(
            "Raster transform is not north-up. This may lead to incorrect results.",
            UserWarning,
        )


@contextmanager
def open_raster(source, *, affine=None, nodata=None, band=1, crs=None):
    """
    Yield a rasterio.DatasetReader no matter what the caller passes.

    Parameters
    ----------
    source   : str | Path | rasterio.io.DatasetReader | np.ndarray
    affine   : Affine, required if `source` is a NumPy array
    nodata   : float, optional override (ignored for open readers that already match)
    band     : int specifying the bad to read
    crs      : dict or str, CRS for in-memory arrays

    Yields
    ------
    ds : rasterio.io.DatasetReaderBase
    """

    memfile = None
    base_ds = None
    close_base = False

    try:
        # already opened reader case
        if isinstance(source, rio.io.DatasetReaderBase):
            base_ds = source
            close_base = False

        elif isinstance(source, (str, bytes, os.PathLike)):
            base_ds = rio.open(source, "r")
            close_base = True

        # in-memory DatasetReader for raw NumPy arrays
        elif isinstance(source, np.ndarray):
            if affine is None:
                raise ValueError("affine must be provided when source is a NumPy array")

            source = np.atleast_2d(source)

            profile = {
                "driver": "GTiff",
                "height": source.shape[0],
                "width": source.shape[1],
                "count": 1,
                "dtype": source.dtype,
                "crs": crs,
                "transform": affine,
                "nodata": nodata,
            }

            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(source, 1)

            base_ds = memfile.open()
            close_base = True
        else:
            raise TypeError("Unsupported raster input type")

        # If custom stuff is specified, create a VRT
        rewrite_nodata = nodata is not None and nodata != base_ds.nodata
        rewrite_crs = crs is not None and crs != base_ds.crs
        rewrite_transform = affine is not None and affine != base_ds.transform
        rewrite_band = band != 1

        create_vrt = rewrite_nodata or rewrite_crs or rewrite_transform or rewrite_band

        if create_vrt:
            vrt_options = {}
            if rewrite_nodata:
                vrt_options["nodata"] = nodata
            if rewrite_crs:
                vrt_options["crs"] = crs
            if rewrite_transform:
                vrt_options["transform"] = affine
            if rewrite_band:
                vrt_options["band_indexes"] = [band]

            ds = WarpedVRT(base_ds, **vrt_options)
            close_ds = True
        else:
            ds = base_ds
            close_ds = False

        yield ds

    finally:
        # Clean up in reverse order
        if "ds" in locals() and close_ds:
            ds.close()

        if "base_ds" in locals() and close_base:
            base_ds.close()

        if memfile is not None:
            memfile.close()


def open_vector(
    vectors: Union[
        str,
        Path,  # file path / URL
        BaseGeometry,  # single Shapely geometry
        Sequence[BaseGeometry],  # list/tuple of geometries
        dict,  # GeoJSON-like mapping
        gpd.GeoDataFrame,
    ],
    *,
    layer: Union[int, str] = 0,
    crs=None,
    affine: Affine | None = None,
) -> gpd.GeoDataFrame:
    """
    Normalise *vectors* into a GeoDataFrame.

    Parameters
    ----------
    vectors   : many flavours, see type hint above.
    layer     : layer name or index when reading multi-layer files.
    crs       : CRS to assign if the input has none (e.g. Shapely geom).
    affine    : Affine of the *raster* (optional).  If crs is None and you
                pass an Affine with EPSG info (pyproj 3.6+ Affine.crs), that
                CRS will be used.

    Returns
    -------
    gdf : GeoDataFrame
        Always contains at least a 'geometry' column and a CRS.
        The caller can attach extra attributes with `gdf.attrs[...]`.
    """

    # GeoDataFrame
    if isinstance(vectors, gpd.GeoDataFrame):
        gdf = vectors.copy()

    # String
    elif isinstance(vectors, (str, Path)):
        try:
            gdf = gpd.read_file(vectors, layer=layer)
            # remember original pathname so later code can emit it if needed
            gdf.attrs["file_path"] = str(vectors)
        except Exception:
            # maybe it's a wkt string?
            try:
                gdf = gpd.GeoDataFrame({"geometry": [wkt.loads(vectors)]}, crs=crs)
            except Exception:
                # maybe it's an unparsed GeoJSON mapping?
                gdf = gpd.GeoDataFrame.from_features(json.loads(vectors), crs=crs)
            except Exception as e:
                raise ValueError(f"Cannot read vector data from {vectors!r}: {e}")

    # dict GeoJSON-like mapping
    elif isinstance(vectors, dict):
        if "type" not in vectors:
            raise ValueError("GeoJSON mapping must have a 'type' key")
        gdf = gpd.GeoDataFrame.from_features(vectors, crs=crs)

    # single geometry or sequence of geometries
    elif isinstance(vectors, BaseGeometry) or (
        isinstance(vectors, (list, tuple))
        and all(isinstance(g, BaseGeometry) for g in vectors)
    ):
        geoms = [vectors] if isinstance(vectors, BaseGeometry) else list(vectors)
        gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)

    # i give up
    else:
        raise TypeError(
            "Unsupported vector input. Must be a file path, GeoDataFrame, "
            "Shapely geometry, GeoJSON mapping, or sequence of geometries."
        )

    if gdf.crs is None:
        if crs is None and affine is not None and hasattr(affine, "crs"):
            crs = affine.crs
        if crs is not None:
            gdf.set_crs(crs, inplace=True)

    return gdf

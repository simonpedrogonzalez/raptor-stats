import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import Window

def to_numpy2(transform):
    return np.array([transform.a, 
    transform.b, 
    transform.c, 
    transform.d, 
    transform.e, 
    transform.f, 0, 0, 1], dtype=np.float32).reshape((3,3))

def xy_np(transform, cols, rows, offset='center'):
    # https://gis.stackexchange.com/questions/415062/how-to-speed-up-rasterio-transform-xy
    # A faster xy trasnform than rasterio.transform.xy
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows, cols, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows
        pts[1] = cols

    if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    _transnp = to_numpy2(transform)
    _translt = to_numpy2(transform.translation(coff, roff))
    locs = _transnp @ _translt @ pts
    return locs[0], locs[1]

def vectorize_raster_to_points(raster: rio.DatasetReader, window: Window = None):
    """Vectorize a raster into a GeoDataFrame of points.
    It does not read the raster, just uses the raster's
    transform and crs to create the points.
    """

    transform = raster.transform
    crs = raster.crs

    if window is not None:
        n_cols = window.width
        n_rows = window.height
        unique_row_indices = np.arange(window.height) + window.row_off
        unique_col_indices = np.arange(window.width) + window.col_off
    else:
        n_cols = raster.width
        n_rows = raster.height
        unique_row_indices = np.arange(n_rows)
        unique_col_indices = np.arange(n_cols)

    assert n_cols * n_rows < 20e6, "Too many points to vectorize"    
    
    row_indices, col_indices = np.meshgrid(
        unique_row_indices.astype(np.int32),
        unique_col_indices.astype(np.int32),
        indexing='ij',
        sparse=False
        )

    row_indices = row_indices.ravel()
    col_indices = col_indices.ravel()

    x, y = xy_np(transform, row_indices, col_indices)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs)

    return points, (row_indices, col_indices)

def split_window_into_quadrants(window: rio.windows.Window):
    half_width = int(window.width // 2)
    half_height = int(window.height // 2)

    wtl = Window(window.col_off, window.row_off, half_width, half_height)  
    wtr = Window(window.col_off + half_width, window.row_off, window.width - half_width, half_height)
    wbl = Window(window.col_off, window.row_off + half_height, half_width, window.height - half_height)
    wbr = Window(window.col_off + half_width, window.row_off + half_height, window.width - half_width, window.height - half_height)

    return wtl, wtr, wbl, wbr


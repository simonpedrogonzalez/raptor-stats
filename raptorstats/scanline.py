import geopandas as gpd
import numpy as np

# Debugging code
# from raptorstats.debugutils import plot_mask_comparison, ref_mask_rasterstats, compare_stats
# import matplotlib.pyplot as plt
# End Debugging code
import rasterio as rio
import shapely
from shapely import LineString, MultiLineString

from raptorstats.zone_stat_method import ZonalStatMethod
from raptorstats.stats import Stats


def xy_to_rowcol(xs, ys, transform):
    cols, rows = (~transform) * (xs, ys)
    return np.floor(rows).astype(int).clip(min=0), np.round(cols).astype(
        int
    ).clip(min=0)

def rows_to_ys(rows, transform):
    return (transform * (0, rows + 0.5))[1]

def build_intersection_table(features: gpd.GeoDataFrame, raster: rio.DatasetReader):
    """Intersect features with scanlines and return the intersection table

    Parameters
    ----------
    features : gpd.GeoDataFrame
    raster : rio.DatasetReader

    Returns
    -------
    (np.ndarray(int), np.ndarray(float))
        f_index, (y, x0, x1) coordinates

    Raises
    ------
    TypeError
        If the intersection type is not LineString or MultiLineString
    """

    window = rio.features.geometry_window(
        raster,
        features.geometry,
        # transform=transform,
        pad_x=0.5,
        pad_y=0.5,
    )

    # Debugging code
    # w_height = int(np.ceil(window.height))
    # w_width = int(np.ceil(window.width))
    # global_mask = np.zeros((w_height+1, w_width), dtype=int)
    # End Debugging code

    row_start, row_end = int(np.floor(window.row_off)), int(
        np.ceil(window.row_off + window.height)
    )
    col_start, col_end = int(np.floor(window.col_off)), int(
        np.ceil(window.col_off + window.width)
    )

    x0 = (raster.transform * (col_start - 1, 0))[0]
    x1 = (raster.transform * (col_end, 0))[0]
    all_rows = np.arange(row_start, row_end + 1)
    ys = rows_to_ys(all_rows, raster.transform)

    x0s = np.full_like(ys, x0, dtype=float)
    x1s = np.full_like(ys, x1, dtype=float)

    # all points defining the scanlines
    all_points = np.stack(
        [np.stack([x0s, ys], axis=1), np.stack([x1s, ys], axis=1)], axis=1
    )

    # for g in features.geometry:
    #     shapely.prepare(g)
    scanlines = MultiLineString(list(all_points))
    # shapely.prepare(scanlines)

    all_intersections = scanlines.intersection(features.geometry)

    intersection_table = []
    for f_index, inter in enumerate(all_intersections):
        if inter.is_empty:
            continue
        if isinstance(inter, MultiLineString):
            geoms = inter.geoms
        elif isinstance(inter, LineString):
            geoms = [inter]
        else:
            raise TypeError(f"Unexpected intersection type: {type(inter)}")

        # g_arr = np.asarray(geoms, dtype=object)
        g_arr = shapely.get_parts(geoms)
        starts = shapely.get_point(g_arr, 0)
        ends = shapely.get_point(g_arr, -1)
        x0s = shapely.get_x(starts)
        x1s = shapely.get_x(ends)
        ys = shapely.get_y(starts)

        # vertically stack the intersections
        # creating a numpy array of shape (n, 4)
        # with f_index, y, x0, x1
        intersection_table.extend(
            zip(np.full_like(ys, f_index, dtype=int), ys, x0s, x1s)
        )

    if not intersection_table:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    intersection_table = np.array(intersection_table)
    f_index = intersection_table[:, 0].astype(int)
    coords = intersection_table[:, 1:4]
    return f_index, coords

def build_reading_table(f_index, intersection_coords, raster: rio.DatasetReader, return_coordinates=False, sort_by_feature=False):
    """Create a reading table indicating which pixels to read for each feature

    Parameters
    ----------
    intersection_table : np.ndarray
        A 2D array containing feature index, y coordinate, x0 coordinate, and x1 coordinate
    raster : rio.DatasetReader
        A rasterio dataset reader object
    return_coordinates : bool, optional
        Whether to return the coordinates of the pixels, by default False
    sort_by_feature : bool, optional
        Whether to sort the reading table by feature index and then by row, instead of just by row, by default False

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        (row, col0, col1, f_index), (y, x0, x1) if return_coordinates is True
    """
    inter_ys = intersection_coords[:, 0]
    inter_x0s = intersection_coords[:, 1]
    inter_x1s = intersection_coords[:, 2]

    rows, col0s = xy_to_rowcol(inter_x0s, inter_ys, raster.transform)
    _, col1s = xy_to_rowcol(inter_x1s, inter_ys, raster.transform)

    reading_table = np.stack([rows, col0s, col1s, f_index], axis=1).astype(int)

    reading_table = reading_table[
        col0s < col1s
    ]  # removes pixel reads where both intersections fall in between pixel centers

    if sort_by_feature:
        # sort by feature first then by row
        sort_idx = np.lexsort((reading_table[:, 0], reading_table[:, 3]))
    else:
        # sort just by row  
        sort_idx = np.argsort(reading_table[:, 0])
    
    if return_coordinates:
        coor = np.stack(
            [inter_ys, inter_x0s, inter_x1s], axis=1
        )[sort_idx] # they are separate cause reading table is integer and coordinates are float

        return reading_table[sort_idx], coor

    return reading_table[sort_idx]

def process_reading_table(reading_table: np.ndarray, features: gpd.GeoDataFrame, raster: rio.DatasetReader, stats: Stats, partials=None):
    """ Read the pixels indicated by the reading table and compute statistics

    Parameters
    ----------
    reading_table : np.ndarray
        A 2D array containing the reading table information
    features : gpd.GeoDataFrame
        A GeoDataFrame containing the features
    raster : rio.DatasetReader
        A rasterio dataset reader object
    stats : Stats
        A Stats object for computing statistics
    partials : list, optional
        A list of partial per-feature statistics to combine with the computed statistics, by default None

    Returns
    -------
    List[Dict]
        A list of statistics for each feature
    """
    rows, row_starts = np.unique(reading_table[:, 0], return_index=True)

    pixel_values_per_feature = [[] for _ in range(len(features.geometry))]

    for i, row in enumerate(rows):
        start = row_starts[i]
        end = row_starts[i + 1] if i + 1 < len(row_starts) else len(reading_table)
        reading_line = reading_table[start:end]

        min_col = np.min(reading_line[:, 1])
        max_col = np.max(reading_line[:, 2])
        reading_window = rio.windows.Window(
            col_off=min_col, row_off=row, width=max_col - min_col, height=1
        )

        # Does not handle nodata
        data = raster.read(1, window=reading_window, masked=True)
        if data.shape[0] == 0:
            continue
        data = data[0]

        for j, col0, col1, f_index in reading_line:

            c0 = col0 - min_col
            c1 = col1 - min_col
            pixel_values = data[c0:c1]

            # Debugging code
            # mark the pixels in the global mask
            # row_in_mask = row - row_start
            # col0_in_mask = col0 - col_start
            # col1_in_mask = col1 - col_start
            # global_mask[row_in_mask, col0_in_mask:col1_in_mask] = f_index + 1
            # End Debugging code

            if len(pixel_values) > 0:
                pixel_values_per_feature[f_index].append(pixel_values)

    # combine the results
    results_per_feature = []
    for i in range(len(pixel_values_per_feature)):
        if not pixel_values_per_feature[i]:
            feature_data = np.ma.array([], mask=True)
        else:
            feature_data = np.ma.concatenate(pixel_values_per_feature[i])
        # get the stats
        r = stats.from_array(feature_data)
        if partials is not None:
            r = stats.from_partials([r, *partials[i]])
        results_per_feature.append(r)

    return results_per_feature


class Scanline(ZonalStatMethod):

    __name__ = "Scanline"

    def __init__(self):
        super().__init__()

    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        # NOTE: ASSUMES NORTH UP, NO SHEAR AFFINE. ASSUMES GEOMETRIES ARE POLYGONS OR MULTIPOLYGONS (NO POINTS, MULTIPOINTS, LINES)
        
        f_index, intersection_coords = build_intersection_table(features, raster)

        if not intersection_coords.size:
            self.results = [
                self.stats.from_array(np.ma.array([], mask=True))
                for _ in features.geometry
            ]
            return

        reading_table = build_reading_table(f_index, intersection_coords, raster, return_coordinates=False, sort_by_feature=False)

        self.results = process_reading_table(reading_table, features, raster, self.stats)
        # Debugging code
        # global_mask = global_mask[:-1, :]  # remove the last row added for debugging
        # ref_mask = ref_mask_rasterstats(features, raster, window)
        # compare_stats(self.results,
        #     self.raster.files[0], features.attrs.get('file_path'), stats=self.stats, show_diff=True, precision=5)
        # plot_mask_comparison(global_mask, ref_mask, features, raster.transform, window=window, scanlines=inter_ys)
        # print('done')
        # end of debugging code

    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        self._precomputations(features, raster)
        return self.results

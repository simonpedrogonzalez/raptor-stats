from raptorstats.zone_stat_method import ZonalStatMethod
# Debugging code
# from raptorstats.debugutils import plot_mask_comparison, ref_mask_rasterstats, compare_stats
# import matplotlib.pyplot as plt
# End Debugging code
import rasterio as rio
import geopandas as gpd
import numpy as np
from shapely import MultiLineString, LineString
import shapely

class Scanline(ZonalStatMethod):

    __name__ = "Scanline"

    def __init__(self):
        super().__init__()

    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        # NOTE: ASSUMES NORTH UP, NO SHEAR AFFINE. ASSUMES GEOMETRIES ARE POLYGONS OR MULTIPOLYGONS (NO POINTS, MULTIPOINTS, LINES)

        transform = raster.transform
        
        def rows_to_ys(rows):
            return (transform * (0, rows + 0.5))[1]

        def xy_to_rowcol(xs, ys):
            cols, rows = (~transform) * (xs, ys)
            return np.floor(rows).astype(int).clip(min=0), np.round(cols).astype(int).clip(min=0)


        window = rio.features.geometry_window(
            raster,
            features.geometry,
            # transform=transform,
            pad_x=0.5, pad_y=0.5,
        )

        # Debugging code
        # w_height = int(np.ceil(window.height))
        # w_width = int(np.ceil(window.width))
        # global_mask = np.zeros((w_height+1, w_width), dtype=int)
        # End Debugging code

        row_start, row_end = int(np.floor(window.row_off)), int(np.ceil(window.row_off + window.height))
        col_start, col_end = int(np.floor(window.col_off)), int(np.ceil(window.col_off + window.width))

        x0 = (raster.transform * (col_start - 1, 0))[0]
        x1 = (raster.transform * (col_end, 0))[0]
        all_rows = np.arange(row_start, row_end + 1)
        ys = rows_to_ys(all_rows)

        x0s = np.full_like(ys, x0, dtype=float)
        x1s = np.full_like(ys, x1, dtype=float)

        # all points defining the scanlines
        all_points = np.stack([
            np.stack([x0s, ys], axis=1),
            np.stack([x1s, ys], axis=1)
        ], axis=1)


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
            intersection_table.extend(zip(
                np.full_like(ys, f_index, dtype=int),
                ys,
                x0s,
                x1s
            ))

        if not intersection_table:
            self.results = [self.stats.from_array(np.ma.array([], mask=True)) for _ in features.geometry]
            return

        intersection_table = np.array(intersection_table)
        inter_x0s = intersection_table[:, 2]
        inter_x1s = intersection_table[:, 3]
        inter_ys = intersection_table[:, 1]
        f_index = intersection_table[:, 0]

        rows, col0s = xy_to_rowcol(inter_x0s, inter_ys)
        _, col1s = xy_to_rowcol(inter_x1s, inter_ys)

        reading_table = np.stack([
            rows, col0s, col1s, f_index
        ], axis=1).astype(int)

        reading_table = reading_table[col0s < col1s] # removes pixel reads where both intersections fall in between pixel centers

        # sort by row
        reading_table = reading_table[np.argsort(reading_table[:, 0])]
        rows, row_starts = np.unique(reading_table[:, 0], return_index=True)

        pixel_values_per_feature = [[] for _ in range(len(all_intersections))]

        for i, row in enumerate(rows):
            start = row_starts[i]
            end = row_starts[i+1] if i+1 < len(row_starts) else len(reading_table)
            reading_line = reading_table[start:end]

            min_col = np.min(reading_line[:, 1])
            max_col = np.max(reading_line[:, 2])
            reading_window = rio.windows.Window(
                col_off=min_col,
                row_off=row,
                width=max_col - min_col,
                height=1
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
            r = self.stats.from_array(feature_data)
            results_per_feature.append(r)

        self.results = results_per_feature

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




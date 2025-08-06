from raptorstats.zone_stat_method import ZonalStatMethod
from raptorstats.debugutils import plot_mask_comparison, ref_mask_rasterstats, compare_stats
import rasterio as rio
import geopandas as gpd
import numpy as np
from shapely import MultiLineString
import matplotlib.pyplot as plt
from rasterio.windows import transform as window_transform   # handy helper
# import line_profiler


class Scanline(ZonalStatMethod):

    __name__ = "Scanline"

    def __init__(self):
        super().__init__()

    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):

        transform = raster.transform

        def rows_to_ys(rows):
            return (transform * (0, rows + 0.5))[1]
        
        def rowcol_to_xy(rows, cols):
            # coords of pixel centers
            xs, ys = (transform * (cols + 0.5, rows + 0.5))
            return xs, ys

        # def ys_to_rows(ys):
        #     return ((~transform) * (np.zeros_like(ys), ys))[1].astype(int)
        
        def xy_to_rowcol(xs, ys):
            # pixel in which the point is located
            cols, rows = (~transform) * (xs, ys)
            return np.floor(rows).astype(int), np.floor(cols).astype(int)

        # get window for the entire features, that is, the bounding box for all features
        # window = rio.windows.from_bounds(
        #     *features.total_bounds,
        #     transform=transform,

        # )
        # # snap to pixel grid
        # window = window.round_offsets(op='floor').round_lengths(op='ceil')

        window = rio.features.geometry_window(
            raster,
            features.geometry,
            # transform=transform,
            pad_x=0.5, pad_y=0.5,
        )

        # Debugging code
        w_height = int(np.ceil(window.height))
        w_width = int(np.ceil(window.width))
        global_mask = np.zeros((w_height+1, w_width), dtype=int)
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

        scanlines = MultiLineString(list(all_points))
        all_intersections = scanlines.intersection(features.geometry)

        intersection_table = []
        for f_index, inter in enumerate(all_intersections):
            for ml in inter.geoms:
                # f_index, y, x0, x1, (space for row, col1,col2) 
                coords = np.asarray(ml.coords)
                y_ = coords[0][1]
                x0 = coords[0][0]
                x1 = coords[-1][0]
                intersection_table.append((
                    f_index, y_, x0, x1
                ))

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
            data = raster.read(1, window=reading_window, masked=False)[0]
            for j, col0, col1, f_index in reading_line:
                c0 = col0 - min_col
                c1 = col1 - min_col
                pixel_values = data[c0:c1]
                
                # Debugging code
                # mark the pixels in the global mask
                row_in_mask = row - row_start
                col0_in_mask = col0 - col_start
                col1_in_mask = col1 - col_start
                global_mask[row_in_mask, col0_in_mask:col1_in_mask] = f_index + 1
                # End Debugging code

                if len(pixel_values) > 0:
                    pixel_values_per_feature[f_index].append(pixel_values)
        

        # combine the results
        results_per_feature = []
        for i in range(len(pixel_values_per_feature)):
            feature_data = np.concatenate(pixel_values_per_feature[i])
            # get the stats
            r = self._compute_stats_from_array(feature_data)
            results_per_feature.append(r)

        self.results = results_per_feature

        # Debugging code
        # from raptorstats.raster_methods import Masking
        # m = Masking()
        # ref_mask = np.zeros_like(global_mask, dtype=int)
        # for i in range(len(features.geometry)):
        #     # get all results for this feature
        #     feature = features.geometry[i]
        #     fmask = m.mask(feature, raster, window)
        #     # add a 0 row at the end of fmask
        #     fmask = np.insert(fmask, 0, 0, axis=0)
        #     ref_mask[fmask] = i+1
        # import matplotlib.pyplot as plt
        # diff_mask = global_mask - ref_mask
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(diff_mask, cmap='inferno')
        # ax.set_title('Difference Mask')
        # plt.show()
        # print('stop')
        global_mask = global_mask[:-1, :]  # remove the last row added for debugging
        ref_mask = ref_mask_rasterstats(features, raster, window)
        compare_stats(self.results,
            self.raster_file_path.files[0], features.attrs.get('file_path'), stats=self.stats, show_diff=True, precision=5)
        
        plot_mask_comparison(global_mask, ref_mask, features, raster.transform, window=window, scanlines=inter_ys)
        print('done')
        # end of debugging code    
        
    
    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        self._precomputations(features, raster)
        return self.results




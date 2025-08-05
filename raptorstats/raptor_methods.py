from zone_stat_method import ZonalStatMethod
import rasterio as rio
import geopandas as gpd
from shapely import Geometry, box
import numpy as np
from shapely import MultiLineString
from rtree import index
from raster_methods import Masking
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import line_profiler
from constants import INDICES_PATH
import os
import shutil
from node import Node

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
        window = rio.windows.from_bounds(
            *features.total_bounds,
            transform=transform
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
            feature_data = np.concatenate(pixel_values_per_feature[i])
            # get the stats
            r = self._compute_stats_from_array(feature_data)
            results_per_feature.append(r)

        self.results = results_per_feature

        # Debugging code
        # from raster_methods import Masking
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
        # plt.savefig('diff_mask.png')
        # plt.close()
        # end of debugging code    
        
    
    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        self._precomputations(features, raster)
        return self.results


class AggQuadTree(ZonalStatMethod):

    
    __name__ = "AggQuadTree"

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        super().__init__()

    def _get_params(self):
        return {
            "max_depth": self.max_depth
        }

    def _get_index_path(self):
        raster_file_name = self.raster_file_path.split("/")[-1]
        raster_base_name = os.path.splitext(raster_file_name)[0]
        max_depth = self.max_depth
        index_path = os.path.join(INDICES_PATH, f"index_{raster_base_name}_depth_{max_depth}")
        return index_path

    def _compute_scanline_reading_table(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):

        transform = raster.transform

        def rows_to_ys(rows):
            return (transform * (0, rows + 0.5))[1]
        
        def rowcol_to_xy(rows, cols):
            # coords of pixel centers
            xs, ys = (transform * (cols + 0.5, rows + 0.5))
            return xs, ys
        
        def xy_to_rowcol(xs, ys):
            # pixel in which the point is located
            cols, rows = (~transform) * (xs, ys)
            return np.floor(rows).astype(int), np.floor(cols).astype(int)

        # get window for the entire features, that is, the bounding box for all features
        window = rio.windows.from_bounds(
            *features.total_bounds,
            transform=transform
        )

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
            inter_ys, inter_x0s, inter_x1s, rows, col0s, col1s, f_index
        ], axis=1).astype(int)
        
        sort_idx = np.lexsort((rows, f_index))  
        reading_table = reading_table[sort_idx]
        unique_f_indices, f_index_starts = np.unique(reading_table[:, 6], return_index=True)

        self._reading_table = reading_table
        self.f_index_starts = f_index_starts
        self.unique_f_indices = unique_f_indices
        self.n_features = len(unique_f_indices)


    @line_profiler.profile
    def _compute_quad_tree(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader):
        
        def part1by1(n):
            n &= 0x0000ffff
            n = (n | (n << 8)) & 0x00FF00FF
            n = (n | (n << 4)) & 0x0F0F0F0F
            n = (n | (n << 2)) & 0x33333333
            n = (n | (n << 1)) & 0x55555555
            return n

        def z_order_indices(xs, ys):
            return (part1by1(ys) << 1) | part1by1(xs)

        def get_boxes(divisions, x_min, x_max, y_min, y_max):
            dx = (x_max - x_min) / divisions
            dy = (y_max - y_min) / divisions
            x_starts = np.linspace(x_min, x_max - dx, divisions)
            y_starts = np.linspace(y_min, y_max - dy, divisions)
            xs, ys = np.meshgrid(x_starts, y_starts)
            boxes = np.stack([xs.flatten(), ys.flatten(), (xs + dx).flatten(), (ys + dy).flatten()], axis=1)

            # ALSO get the corresponding grid indices (ix, iy) to calculate Z-order
            grid_xs, grid_ys = np.meshgrid(np.arange(divisions), np.arange(divisions))
            ix = grid_xs.flatten()
            iy = grid_ys.flatten()

            return boxes, ix, iy

        x_min, y_min, x_max, y_max = raster.bounds
        width = x_max - x_min
        height = y_max - y_min

        n_total_boxes = sum([4 ** i for i in range(self.max_depth + 1)])
        box_index = 0
        idx = index.Index(self._get_index_path())

        all_nodes = []

        for level in range(self.max_depth, -1, -1):
            divisions = 2 ** level
            
            boxes, ix, iy = get_boxes(divisions, x_min, x_max, y_min, y_max)

            # Now compute the Morton Z-order values correctly
            z_indices = z_order_indices(ix, iy)
            # Sort the boxes accordingly
            sort_idx = np.argsort(z_indices)
            boxes = boxes[sort_idx]
            z_indices = z_indices[sort_idx]
            ids = z_indices + box_index
            shp_boxes = [box(x0, y0, x1, y1) for x0, y0, x1, y1 in boxes]
            n_boxes = len(boxes)
            box_index = box_index + n_boxes

            if level == 0:
                parent_ids = np.array([-1])
            else:
                parent_div = 2 ** (level - 1)
                parent_ids = (z_indices // 4) + box_index
            
            if level == self.max_depth:
                
                vector_layer = gpd.GeoDataFrame(
                    geometry=shp_boxes,
                    crs=feature.crs
                )

                # count number of pixels in first box
                box1 = vector_layer.geometry[0]
                win = rio.windows.from_bounds(
                    box1.bounds[0],
                    box1.bounds[1],
                    box1.bounds[2],
                    box1.bounds[3],
                    transform=raster.transform
                )
                data = raster.read(window=win, masked=True)
                data = data[0]
                data = data[~data.mask]

                masking_method = Masking()
                aggregates = masking_method(raster, vector_layer, self.stats)
                aggregates = np.array(aggregates)
                # print(f"Aggregates count: {aggregates[0]['count']}")
            else:
                # use self._combine_stats to combine the stats of the previous level
                # run combine over the previous_aggregates using previous_ids
                # to know which previous_aggregates belong to which parent

                aggregates = []
                unique_previous_parents = np.unique(previous_parent_ids)
                for p_id in unique_previous_parents:
                    mask = previous_parent_ids == p_id
                    child_aggregates = previous_aggregates[mask]
                    combined = self._combine_stats(child_aggregates)
                    aggregates.append(combined)
                aggregates = np.array(aggregates)

            previous_aggregates = aggregates
            previous_parent_ids = parent_ids

            # test_plot(boxes, aggregates, parent_ids, ids)

            ns = []
            for i in range(len(boxes)):
                
                node = Node(
                    _id=ids[i],
                    parent_id=parent_ids[i],
                    level=level,
                    max_depth=self.max_depth,
                    _box=shp_boxes[i],
                    stats=aggregates[i],
                )
                ns.append(node)
                idx.insert(ids[i], boxes[i], obj=node)
                all_nodes.append(node)
        

        self.idx = idx



    def _load_quad_tree(self):
        idx_path = self._get_index_path()
        idx_file = idx_path + ".idx"
        dat_file = idx_path + ".dat"
        if os.path.exists(idx_file) and os.path.exists(dat_file):
            self.idx = index.Index(idx_path)
            return True
        else:
            return False

    @line_profiler.profile
    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        if not self._load_quad_tree():
            self._compute_quad_tree(features, raster)
        self._compute_scanline_reading_table(features, raster)

    @line_profiler.profile
    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        self._precomputations(features, raster)

        reading_table = self._reading_table
        f_index_starts = self.f_index_starts
        unique_f_indices = self.unique_f_indices
        n_features = self.n_features
        transform = raster.transform

        def xy_to_rowcol(xs, ys):
            # pixel in which the point is located
            cols, rows = (~transform) * (xs, ys)
            return np.floor(rows).astype(int), np.floor(cols).astype(int)

        new_reading_table = []
        partials = [[] for _ in range(n_features)]

        for f_index, geom in enumerate(features.geometry):

            # Get all intersecting nodes, sorted from root to leaves
            # window for geom
            window_bounds = geom.bounds
            # window_bounds = rio.windows.bounds(window, raster.transform)
            nodes = list(self.idx.intersection(window_bounds, objects=True))[::-1]
            nodes = [node.object for node in nodes]
            nodes = [node for node in nodes if node.is_contained_in_geom(geom)]
            nodes = sorted(nodes, key=lambda x: x.level)
            without_children = []
            nids = set([int(n.id) for n in nodes])
            for n in nodes:
                if int(n.parent_id) not in nids:
                    without_children.append(n)

            nodes = without_children
            # node_copy = nodes.copy()

            f_index_start = f_index_starts[f_index]
            f_index_end = f_index_starts[f_index + 1] if f_index + 1 < n_features else len(reading_table)
            f_reading_table = reading_table[f_index_start:f_index_end]

            if len(nodes) == 0:
                new_reading_table.append(f_reading_table)
                continue
            
            # if f_index == 25:
            #     nnodes = [n for n in nodes if n.is_contained_in_geom(geom)]
            #     bbx = [n.box for n in nnodes]
            #     iids = [n.id for n in nnodes]
            #     aags = []
            #     piids = [n.parent_id for n in nnodes]
            #     # test_plot(bbx, aags, piids, iids)
            #     print(f"Feature index: {f_index}")
            
            to_read = []
            
            for node in nodes:
            
                # if f_index == 25:
                #     print("Node: " + str(node.id))
                #     print("Parent: " + str(node.parent_id))
                #     print("Level: " + str(node.level))
                #     print("Box: " + str(node.box.bounds))
                #     print("Box: " + str((np.array(node.box.bounds) / 1e6).round(3)))

                bounds = node.box.bounds
                n_x0, n_y0, n_x1, n_y1 = bounds
                
                partials[f_index].append(node.stats)

                ys = f_reading_table[:, 0]
                x0s = f_reading_table[:, 1]
                x1s = f_reading_table[:, 2]

                inside_y = (ys >= n_y0) & (ys <= n_y1)
                inside_x = (x0s >= n_x0) & (x1s <= n_x1)
                fully_inside = inside_y & inside_x

                f_reading_table = f_reading_table[~fully_inside]
                ys = f_reading_table[:, 0]
                x0s = f_reading_table[:, 1]
                x1s = f_reading_table[:, 2]
                rows = f_reading_table[:, 3]
                col0s = f_reading_table[:, 4]
                col1s = f_reading_table[:, 5]
                f_idxs = f_reading_table[:, 6]
                # y, x0, x1, rows, col0s, col1s, f_index

                inside_y = (ys >= n_y0) & (ys <= n_y1)
                overlaps_left = (x0s < n_x0) & (x1s > n_x0)  # crosses left edge
                overlaps_right = (x1s > n_x1) & (x0s < n_x1) # crosses right edge
                overlaps_both = (x0s < n_x0) & (x1s > n_x1)   # crosses both sides
                horizontally_intersect = (x0s < n_x1) & (x1s > n_x0)

                new_entries = []

                cut_left = horizontally_intersect & (overlaps_left | overlaps_both) & inside_y
                n_cut_left = np.sum(cut_left)
                if n_cut_left > 0:
                    new_entry_left = np.stack([
                        ys[cut_left],
                        x0s[cut_left],  # new x0 is the left border
                        np.ones(n_cut_left) * n_x0,
                        rows[cut_left],
                        -1 * np.ones(n_cut_left),  # col0 placeholder
                        -1 * np.ones(n_cut_left),  # col1 placeholder
                        f_idxs[cut_left]
                    ], axis=1)
                    new_entries.append(new_entry_left)
                
                cut_right = horizontally_intersect & (overlaps_right | overlaps_both) & inside_y
                n_cut_right = np.sum(cut_right)
                if n_cut_right > 0:
                    new_entry_right = np.stack([
                        ys[cut_right],
                        np.ones(n_cut_right) * n_x1,  # new x0 is right border
                        x1s[cut_right],
                        rows[cut_right],
                        -1 * np.ones(n_cut_right),  # col0 placeholder
                        -1 * np.ones(n_cut_right),  # col1 placeholder
                        f_idxs[cut_right]
                    ], axis=1)
                    new_entries.append(new_entry_right)
                
                completely_outside = ~inside_y | ~horizontally_intersect
                outside_rows = f_reading_table[completely_outside]

                if new_entries:
                    new_entries = np.vstack(new_entries)
                    new_entries_ys = new_entries[:, 0]
                    new_entries_x0s = new_entries[:, 1]
                    new_entries_x1s = new_entries[:, 2]
                    _, new_entries_col0s = xy_to_rowcol(new_entries_x0s, new_entries_ys)
                    _, new_entries_col1s = xy_to_rowcol(new_entries_x1s, new_entries_ys)
                    new_entries[:, 4] = new_entries_col0s
                    new_entries[:, 5] = new_entries_col1s
                    f_reading_table = np.vstack([outside_rows, new_entries])
                else:
                    f_reading_table = outside_rows

                # if f_index == 25:
                #     print(25)
            
            new_reading_table.append(f_reading_table)
            

        new_reading_table = np.vstack(new_reading_table)
        # sort new reading table by rows
        sort_idx = np.lexsort((new_reading_table[:, 3], new_reading_table[:, 6]))
        reading_table = new_reading_table[sort_idx]
        reading_table = reading_table[:, 3:].astype(int)
        rows, row_starts = np.unique(reading_table[:, 3], return_index=True)

        pixel_values_per_feature = [[] for _ in range(n_features)]
        
    

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

                if len(pixel_values) > 0:
                    pixel_values_per_feature[f_index].append(pixel_values)
        
        # combine the results
        results_per_feature = []
        for i in range(len(pixel_values_per_feature)):
            feature_data = np.concatenate(pixel_values_per_feature[i])
            # get the stats
            r = self._compute_stats_from_array(feature_data)
            tree_r = partials[i]
            tree_r.append(r)
            r = self._combine_stats(tree_r)
            results_per_feature.append(r)

        self.results = results_per_feature

        return self.results



def test_plot(boxes, aggregates, parent_ids, ids):

    nx, ny = 8, 8
    # normalize boxes to fit in the figure
    x_min = min(boxes[:, [0, 2]].flatten())
    x_max = max(boxes[:, [0, 2]].flatten())
    y_min = min(boxes[:, [1, 3]].flatten())
    y_max = max(boxes[:, [1, 3]].flatten())

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_min) / (x_max - x_min) * nx
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_min) / (y_max - y_min) * ny



    fig, ax = plt.subplots(figsize=(nx, ny))

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Label the box with its index at the center
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        tt = aggregates[i]['count']
        pp = parent_ids[i]

        color = 'black'
        if int(pp) in [1330, 1220, 1221, 1224]:
            color = 'red'

        ii = ids[i]
        ttt = str(ii) + ' p:' + str(pp)
        ax.text(cx, cy, ttt, fontsize=8, ha='center', va='center', color=color)

    ax.set_xlim(min(boxes[:, 0]), max(boxes[:, 2]))
    ax.set_ylim(min(boxes[:, 1]), max(boxes[:, 3]))
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Optional: to mimic raster image top-down layout
    plt.grid(True)
    plt.show()
    plt.savefig('test_plot.png')


def plot_masks(raster, geom, my_mask, window):

    correct_mask = Masking().mask(geom, raster, window)
    # plot 3 things:
    # my_mask
    # correct_mask
    # difference between the two masks
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(my_mask, cmap='gray')
    ax[0].set_title('My Mask')
    ax[1].imshow(correct_mask, cmap='gray')
    ax[1].set_title('Correct Mask')
    ax[2].imshow(my_mask.astype(int) - correct_mask.astype(int), cmap='gray')
    ax[2].set_title('Difference')
    plt.show()
    print('stop')
    # add colorbar to the first two plots
    
    # plot difference between the two masks
    # mask - correct_mask
    # print('done')

    


def test_raster():

    # 10x10 raster with all 1s
    array = np.ones((8, 8), dtype=np.uint8)
    transform = rio.transform.from_origin(0, 8, 1, 1)  # (x_min, y_max, x_res, y_res)
    from constants import RASTER_DATA_PATH


    # Create the GeoTIFF file
    with rio.open(
        f"{RASTER_DATA_PATH}/test.tif",
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",  # WGS84
        transform=transform,
    ) as dst:
        dst.write(array, 1)

# def test():

#     from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH

#     ag = AggQuadTree(max_depth=3)
#     vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
#     raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR.tif'

#     test_raster()
#     raster_layer_file = f'{RASTER_DATA_PATH}/test.tif'
#     ag(raster_layer_file, vector_layer_file, ['count', 'mean'])


def test():
    from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH
    from reference import reference_method
    ag = AggQuadTree(max_depth=7)
    vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
    raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif'

    # test_raster()
    # raster_layer_file = f'{RASTER_DATA_PATH}/test.tif'
    r1 = ag(raster_layer_file, vector_layer_file, ['count'])

    rlow, rhigh = reference_method(raster_layer_file, vector_layer_file, ['count'])

    for i in range(len(r1)):
        r_c = r1[i]['count']
        r_l = rlow[i]['count']
        r_h = rhigh[i]['count']
        err1 = abs(r_c - r_l) / r_l
        err2 = abs(r_c - r_h) / r_h
        print(f"Error {i}: {err1:.2%}")
        if err1 > 0.03:
            print(f"r_c: {r_c}, r_l: {r_l}, r_h: {r_h}")



    print('done')

# test()
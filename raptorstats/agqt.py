import os

import geopandas as gpd
import line_profiler
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from raptorstats.node import Node
from raptorstats.raster_methods import Masking
from raptorstats.scanline import Scanline
from rtree import index
from shapely import Geometry, MultiLineString, box
from raptorstats.zone_stat_method import ZonalStatMethod
from raptorstats.stats import Stats
from raptorstats.scanline import build_intersection_table, build_reading_table, process_reading_table
import warnings

class AggQuadTree(ZonalStatMethod):

    __name__ = "AggQuadTree"

    def __init__(self, max_depth: int = 5, index_path: str = None, build_index: bool = False):
        self.max_depth = max_depth
        if index_path is None:
            index_path = f"index_{self.__name__}_depth_{max_depth}"
        else:
            # strip .{shit} if exists
            index_path = index_path.split(".")[0]
        self.index_path = index_path
        self.build_index = build_index
        super().__init__()

    def _index_files_exist(self):
        idx_file = f"{self.index_path}.idx"
        dat_file = f"{self.index_path}.dat"
        return os.path.exists(idx_file) and os.path.exists(dat_file)

    def _should_build_indices(self):
        return self.build_index or not self._index_files_exist()

    def _get_params(self):
        return {"max_depth": self.max_depth, "index_path": self.index_path}

    # def _get_index_path(self):
        # raster_file_name = self.raster_file_path.split("/")[-1]
        # raster_base_name = os.path.splitext(raster_file_name)[0]
        # max_depth = self.max_depth
    
        # index_path = os.path.join(
            # self.index_path, f"index_{raster_base_name}_depth_{max_depth}"
        # )
        # return index_path

    def _compute_scanline_reading_table(
        self, features: gpd.GeoDataFrame, raster: rio.DatasetReader
    ):

        # transform = raster.transform

        # def rows_to_ys(rows):
        #     return (transform * (0, rows + 0.5))[1]

        # def rowcol_to_xy(rows, cols):
        #     # coords of pixel centers
        #     xs, ys = transform * (cols + 0.5, rows + 0.5)
        #     return xs, ys

        # def xy_to_rowcol(xs, ys):
        #     # pixel in which the point is located
        #     cols, rows = (~transform) * (xs, ys)
        #     return np.floor(rows).astype(int), np.floor(cols).astype(int)

        # # get window for the entire features, that is, the bounding box for all features
        # window = rio.windows.from_bounds(*features.total_bounds, transform=transform)

        # row_start, row_end = int(np.floor(window.row_off)), int(
        #     np.ceil(window.row_off + window.height)
        # )
        # col_start, col_end = int(np.floor(window.col_off)), int(
        #     np.ceil(window.col_off + window.width)
        # )

        # x0 = (raster.transform * (col_start - 1, 0))[0]
        # x1 = (raster.transform * (col_end, 0))[0]
        # all_rows = np.arange(row_start, row_end + 1)
        # ys = rows_to_ys(all_rows)

        # x0s = np.full_like(ys, x0, dtype=float)
        # x1s = np.full_like(ys, x1, dtype=float)

        # # all points defining the scanlines
        # all_points = np.stack(
        #     [np.stack([x0s, ys], axis=1), np.stack([x1s, ys], axis=1)], axis=1
        # )

        # scanlines = MultiLineString(list(all_points))
        # all_intersections = scanlines.intersection(features.geometry)

        # intersection_table = []
        # for f_index, inter in enumerate(all_intersections):
        #     for ml in inter.geoms:
        #         # f_index, y, x0, x1, (space for row, col1,col2)
        #         coords = np.asarray(ml.coords)
        #         y_ = coords[0][1]
        #         x0 = coords[0][0]
        #         x1 = coords[-1][0]
        #         intersection_table.append((f_index, y_, x0, x1))

        # intersection_table = np.array(intersection_table)
        # inter_x0s = intersection_table[:, 2]
        # inter_x1s = intersection_table[:, 3]
        # inter_ys = intersection_table[:, 1]
        # f_index = intersection_table[:, 0]

        # rows, col0s = xy_to_rowcol(inter_x0s, inter_ys)
        # _, col1s = xy_to_rowcol(inter_x1s, inter_ys)
        # reading_table = np.stack(
        #     [inter_ys, inter_x0s, inter_x1s, rows, col0s, col1s, f_index], axis=1
        # ).astype(int)

        # sort_idx = np.lexsort((rows, f_index))
        # reading_table = reading_table[sort_idx]
        # unique_f_indices, f_index_starts = np.unique(
        #     reading_table[:, 6], return_index=True
        # )
        f_index, intersection_coords = build_intersection_table(features, raster)

        if intersection_coords.size == 0:
            reading_table = np.empty((0, 7), dtype=int)
            coord_table = np.empty((0, 3), dtype=float)
            f_index_starts = np.array([], dtype=int)
            unique_f_indices = np.array([], dtype=int)
            self._reading_table = reading_table
            self._coord_table = coord_table
            self.f_index_starts = f_index_starts
            self.unique_f_indices = unique_f_indices
            self.n_features = len(features.geometry)
            self.effective_n_features = 0
            return

        reading_table, coord_table = build_reading_table(
            f_index, intersection_coords, raster, return_coordinates=True, sort_by_feature=True
        )

        self._reading_table = reading_table # row, col0, col1, f_index
        self._coord_table = coord_table # y, x0, x1
        unique_f_indices, f_index_starts = np.unique(reading_table[:, 3], return_index=True)
        self.f_index_starts = f_index_starts
        self.unique_f_indices = unique_f_indices
        self.n_features = len(features.geometry)
        self.effective_n_features = len(f_index_starts)

    # @line_profiler.profile
    def _compute_quad_tree(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader):

        def part1by1(n):
            n &= 0x0000FFFF
            n = (n | (n << 8)) & 0x00FF00FF
            n = (n | (n << 4)) & 0x0F0F0F0F
            n = (n | (n << 2)) & 0x33333333
            n = (n | (n << 1)) & 0x55555555
            return n

        def z_order_indices(xs, ys):
            return (part1by1(ys) << 1) | part1by1(xs)

        def get_boxes_aligned_even(divisions: int, raster: rio.DatasetReader):
            """
            Build a divisions x divisions grid of pixel-aligned tiles with remainder
            distributed as evenly as possible. Returns (boxes, ix, iy) with boxes
            as (left, bottom, right, top) in CRS coords.
            """
            from rasterio.windows import Window
            from rasterio.windows import bounds as win_bounds

            if divisions <= 0:
                raise ValueError("divisions must be > 0")

            W, H = raster.width, raster.height

            # Integer edges: floor(i*W/D), floor(j*H/D)
            col_edges = (np.arange(divisions + 1) * W) // divisions
            row_edges = (np.arange(divisions + 1) * H) // divisions

            col_offs = col_edges[:-1].astype(int)
            row_offs = row_edges[:-1].astype(int)
            widths   = np.diff(col_edges).astype(int)
            heights  = np.diff(row_edges).astype(int)

            boxes, ix, iy = [], [], []
            for iy_ in range(divisions):
                h = heights[iy_]
                r0 = row_offs[iy_]
                for ix_ in range(divisions):
                    w = widths[ix_]
                    c0 = col_offs[ix_]
                    win = Window(c0, r0, w, h)
                    boxes.append(win_bounds(win, raster.transform))
                    ix.append(ix_)
                    iy.append(iy_)

            return np.asarray(boxes, float), np.asarray(ix, int), np.asarray(iy, int)

        def coarsen_from_children(boxes, ix, iy, divisions):
            """Given current level (divisions×divisions) tiles, build parent level ((div/2)×(div/2))
            by grouping children with parent indices (ix//2, iy//2) and unioning bounds."""
            parent_div = divisions // 2
            pix = ix // 2
            piy = iy // 2
            # Composite key to group by parent cell
            keys = piy * parent_div + pix
            order = np.argsort(keys)
            boxes = boxes[order]; pix = pix[order]; piy = piy[order]; keys = keys[order]

            uniq, starts = np.unique(keys, return_index=True)
            out_boxes = np.empty((len(uniq), 4), dtype=boxes.dtype)
            out_ix = np.empty(len(uniq), dtype=int)
            out_iy = np.empty(len(uniq), dtype=int)

            for k in range(len(uniq)):
                s = starts[k]
                e = starts[k+1] if k+1 < len(uniq) else len(keys)
                g = boxes[s:e]  # children group (≤4 tiles; at edges could be <4 if div was odd, but here div is power of 2)
                # union bounds
                out_boxes[k, 0] = np.min(g[:, 0])  # left
                out_boxes[k, 1] = np.min(g[:, 1])  # bottom
                out_boxes[k, 2] = np.max(g[:, 2])  # right
                out_boxes[k, 3] = np.max(g[:, 3])  # top
                out_ix[k] = pix[s]
                out_iy[k] = piy[s]
            return out_boxes, out_ix, out_iy
        # def get_boxes_intermediate()

        x_min, y_min, x_max, y_max = raster.bounds

        n_pixels_width = raster.width
        n_pixels_height = raster.height
        min_size = min(n_pixels_width, n_pixels_height)
        if min_size / (2**self.max_depth) < 1:
            raise ValueError(
                f"Max depth {self.max_depth} is too high for raster size {n_pixels_width}x{n_pixels_height}. "
                "Reduce max_depth."
            )

        box_index = 0
        idx = index.Index(self.index_path)
        all_nodes = []

        # Build **leaf** level once
        level = self.max_depth
        divisions = 2**level
        boxes, ix, iy = get_boxes_aligned_even(divisions, raster)

        previous_aggregates = None
        previous_parent_ids = None

        # Iterate from leaves up to root, coarsening each time
        for level in range(self.max_depth, -1, -1):
            divisions = 2**level

            # z-order + stable sort for this level
            z_indices = z_order_indices(ix, iy)
            sort_idx = np.argsort(z_indices)
            boxes_lvl = boxes[sort_idx]
            ix_lvl    = ix[sort_idx]
            iy_lvl    = iy[sort_idx]
            z_lvl     = z_indices[sort_idx]

            ids = z_lvl + box_index
            shp_boxes = [box(x0, y0, x1, y1) for (x0, y0, x1, y1) in boxes_lvl]
            n_boxes = len(boxes_lvl)
            box_index += n_boxes  # advance global id space to parent level

            if level == 0:
                parent_ids = np.array([-1])
            else:
                parent_ids = (z_lvl // 4) + box_index  # parent ids live in the *next* chunk

            if level == self.max_depth:
                # Compute aggregates only once (on leaves)
                vector_layer = gpd.GeoDataFrame(geometry=shp_boxes, crs=feature.crs)
                sc = Scanline()
                aggregates = np.array(sc(raster, vector_layer, self.stats))
            else:
                # Combine child aggregates into parents using parent_ids computed at previous (finer) level
                aggregates = []
                for p_id in np.unique(previous_parent_ids):
                    mask = previous_parent_ids == p_id
                    child_aggregates = previous_aggregates[mask]
                    aggregates.append(self.stats.from_partials(child_aggregates))
                aggregates = np.array(aggregates)

            previous_aggregates = aggregates
            previous_parent_ids = parent_ids

            # Emit nodes for this level
            for i in range(n_boxes):
                node = Node(
                    _id=ids[i],
                    parent_id=parent_ids[i] if level > 0 else -1,
                    level=level,
                    max_depth=self.max_depth,
                    _box=shp_boxes[i],
                    transform=raster.transform,
                    stats=aggregates[i],
                )
                idx.insert(ids[i], boxes_lvl[i], obj=node)
                all_nodes.append(node)

            # Prepare **parent** level boxes by unioning children bounds
            if level > 0:
                boxes, ix, iy = coarsen_from_children(boxes_lvl, ix_lvl, iy_lvl, divisions)
        self.idx = idx

    def _load_quad_tree(self):
        try:
            self.idx = index.Index(self.index_path)
        except Exception as e:
            print(f"Error loading index: {e}")
            raise e

    def _delete_index_files(self):
        idx_file = f"{self.index_path}.idx"
        dat_file = f"{self.index_path}.dat"
        if os.path.exists(idx_file):
            os.remove(idx_file)
        if os.path.exists(dat_file):
            os.remove(dat_file)

    # @line_profiler.profile
    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        if self._should_build_indices():
            if self._index_files_exist():
                warnings.warn(
                    f"Index files {self.index_path}.idx and {self.index_path}.dat already exist. They will be overwritten.",
                )
                self._delete_index_files()
            self._compute_quad_tree(features, raster)
        else:
            self._load_quad_tree()
        self._compute_scanline_reading_table(features, raster)

    # @line_profiler.profile
    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        self._precomputations(features, raster)

        reading_table = self._reading_table
        coord_table = self._coord_table
        f_index_starts = self.f_index_starts
        n_features = self.n_features
        n_eff_features = self.effective_n_features

        if reading_table.size == 0:
            # No intersections found, return empty stats
            return [self.stats.empty() for _ in range(n_features)]
        
        # transform = raster.transform

        # def xy_to_rowcol(xs, ys, transform):
        #     cols, rows = (~transform) * (xs, ys)

        #     rows = np.floor(rows).astype(int).clip(min=0)
        #     cols = np.ceil(cols - 0.5).astype(int).clip(min=0)   # left-inclusive
        #     # cols = np.round(cols).astype(int).clip(min=0)
        #     return rows, cols

        new_reading_table = []
        new_coord_table = []
        partials = [[] for _ in range(n_features)]

        # Debugging
        # H, W = raster.height, raster.width
        # window = rio.windows.Window(0, 0, W, H).round_offsets().round_lengths()  # for plotting/ref
        # global_mask = np.zeros((H, W), dtype=int)  # global mask for debugging
        # used_nodes = [[] for _ in range(n_features)]
        # End Debugging

        for f_index, geom in enumerate(features.geometry):

            if f_index not in self.unique_f_indices:
                # skip features that do not have any intersections
                # self.unique_f_indices was computed from the scanline reading table
                # so if this f_index does not appear, it means there are n intersections
                continue

            # Get all intersecting nodes, sorted from root to leaves
            # window for geom
            window_bounds = geom.bounds
            # window_bounds = rio.windows.bounds(window, raster.transform)
            nodes = list(self.idx.intersection(window_bounds, objects=True))[::-1]
            nodes = [node.object for node in nodes] # TODO: one liner
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
            f_index_end = (
                f_index_starts[f_index + 1]
                if f_index + 1 < n_eff_features
                else len(reading_table)
            )
            f_reading_table = reading_table[f_index_start:f_index_end]
            f_coord_table = coord_table[f_index_start:f_index_end]

            if len(nodes) == 0:
                new_reading_table.append(f_reading_table)
                new_coord_table.append(f_coord_table)
                continue

            # Debugging
            # put node 1 in position 0
            # n1 = nodes[0]
            # n0 = nodes[0]
            # nodes[0] = nodes[1]
            # nodes[1] = n0
            # End Debugging

            for node in nodes:

                # if f_index == 25:
                #     print("Node: " + str(node.id))
                #     print("Parent: " + str(node.parent_id))
                #     print("Level: " + str(node.level))
                #     print("Box: " + str(node.box.bounds))
                #     print("Box: " + str((np.array(node.box.bounds) / 1e6).round(3)))

                # bounds = node.box.bounds
                # n_x0, n_y0, n_x1, n_y1 = bounds
                n_r0, n_r1, n_c0, n_c1 = node.pixel_bounds
                # Debugging
                # Mark in global mask

                # sq_win = rio.windows.from_bounds(
                #     n_x0, n_y0, n_x1, n_y1, transform=transform
                # )
                # sq_win = sq_win.round_offsets().round_lengths()
                # inter = rio.windows.intersection(sq_win, window)
                # # mark in global mask
                # loc_row0 = int(round(inter.row_off - subset_row0))
                # loc_col0 = int(round(inter.col_off - subset_col0))
                # h = int(round(inter.height))
                # w = int(round(inter.width))
                # r1 = max(0, loc_row0); c1 = max(0, loc_col0)
                # r2 = min(global_mask.shape[0], r1 + h)
                # c2 = min(global_mask.shape[1], c1 + w)
                # if r1 < r2 and c1 < c2:
                #     global_mask[r1:r2, c1:c2] = f_index + 1
                # used_nodes[f_index].append(node)

                # ensure stats are correct
                # import rasterstats
                # assert rasterstats.zonal_stats(node.box, raster.name, stats="count")[0]['count'] == node.stats['count']

                # End Debugging

                partials[f_index].append(node.stats)

                # ys = f_coord_table[:, 0]
                # x0s = f_coord_table[:, 1]
                # x1s = f_coord_table[:, 2]

                # inside_y = (ys >= n_y0) & (ys <= n_y1)
                # inside_x = (x0s >= n_x0) & (x1s <= n_x1)
                # fully_inside = inside_y & inside_x

                rows = f_reading_table[:, 0]
                col0s = f_reading_table[:, 1]
                col1s = f_reading_table[:, 2]
                fully_inside = (rows >= n_r0) & (rows < n_r1) & (col0s >= n_c0) & (col1s <= n_c1)

                f_reading_table = f_reading_table[~fully_inside]
                # f_coord_table = f_coord_table[~fully_inside]

                # ys = f_coord_table[:, 0]
                # x0s = f_coord_table[:, 1]
                # x1s = f_coord_table[:, 2]
                rows = f_reading_table[:, 0]
                col0s = f_reading_table[:, 1]
                col1s = f_reading_table[:, 2]

                f_idxs = f_reading_table[:, 3]
                # y, x0, x1, rows, col0s, col1s, f_index

                # inside_y = (rows >= n_r0) & (rows < n_r1)
                # horizontally_intersect = (col0s < n_c1) & (col1s > n_c0)
                # overlaps_left  = inside_y & (col0s < n_c0) & (col1s > n_c0) & (col1s <= n_c1)
                # overlaps_right = inside_y & (col0s >= n_c0) & (col0s < n_c1) & (col1s > n_c1)
                # overlaps_both  = inside_y & (col0s < n_c0) & (col1s > n_c1)  # spans across both edges

                new_entries = []

                # cut_left = (
                #     horizontally_intersect & (overlaps_left | overlaps_both) & inside_y
                # )
                # n_cut_left = np.sum(cut_left)
                # if n_cut_left > 0:
                #     new_entry_left = np.stack(
                #         [
                #             ys[cut_left], # y
                #             x0s[cut_left],  # new x0 is the left border
                #             np.ones(n_cut_left) * n_x0,
                #             rows[cut_left],
                #             -1 * np.ones(n_cut_left),  # col0 placeholder
                #             -1 * np.ones(n_cut_left),  # col1 placeholder
                #             f_idxs[cut_left],
                #         ],
                #         axis=1,
                #     )
                #     # new_entries.append(new_entry_left)
                
                Lmask = (rows >= n_r0) & (rows < n_r1) & (col0s < n_c0) & (col1s > n_c0)
                if np.any(Lmask):
                        l_rows = rows[Lmask]
                        l_c0   = col0s[Lmask]
                        l_c1   = np.minimum(col1s[Lmask], n_c0)  # cut at left edge
                        keep   = l_c1 > l_c0                     # drop zero-length
                        if np.any(keep):
                            left_reading = np.stack(
                                [l_rows[keep], l_c0[keep], l_c1[keep], f_idxs[Lmask][keep]],
                                axis=1
                            ).astype(int)
                            new_entries.append(left_reading)

                # cut_right = (
                #     horizontally_intersect & (overlaps_right | overlaps_both) & inside_y
                # )
                # n_cut_right = np.sum(cut_right)
                # if n_cut_right > 0:
                #     new_entry_right = np.stack(
                #         [
                #             ys[cut_right],
                #             np.ones(n_cut_right) * n_x1,  # new x0 is right border
                #             x1s[cut_right],
                #             rows[cut_right],
                #             -1 * np.ones(n_cut_right),  # col0 placeholder
                #             -1 * np.ones(n_cut_right),  # col1 placeholder
                #             f_idxs[cut_right],
                #         ],
                #         axis=1,
                #     )
                #     # TODO: uncomment
                #     # new_entries.append(new_entry_right)

                # Right cut: segments crossing the node's right edge
                Rmask = (rows >= n_r0) & (rows < n_r1) & (col1s > n_c1) & (col0s < n_c1)

                if np.any(Rmask):
                    r_rows = rows[Rmask]
                    r_c0   = np.maximum(col0s[Rmask], n_c1)  # start at node's right edge
                    r_c1   = col1s[Rmask]
                    r_f    = f_idxs[Rmask]

                    keep = r_c1 > r_c0  # drop zero-length pieces
                    if np.any(keep):
                        right_reading = np.stack(
                            [r_rows[keep], r_c0[keep], r_c1[keep], r_f[keep]],
                            axis=1
                        ).astype(int)
                        new_entries.append(right_reading)


                # TODO: uncomment
                # completely_outside = ~inside_y | ~horizontally_intersect
                # outside_rows = f_reading_table[completely_outside]
                # outside_coords = f_coord_table[completely_outside]

                # if new_entries:
                    
                #     new_entries = np.vstack(new_entries)
                #     new_entries_ys = new_entries[:, 0]
                #     new_entries_x0s = new_entries[:, 1]
                #     new_entries_x1s = new_entries[:, 2]
                #     new_entries_f_index = new_entries[:, 6]
                #     new_entries_rows = new_entries[:, 3]

                #     new_coord_entries = np.stack([
                #         new_entries_ys,
                #         new_entries_x0s,
                #         new_entries_x1s,
                #     ], axis=1)
                    
                #     _, new_entries_col0s = xy_to_rowcol(new_entries_x0s, new_entries_ys, raster.transform)
                #     _, new_entries_col1s = xy_to_rowcol(new_entries_x1s, new_entries_ys, raster.transform)
                    
                #     new_reading_table_entries = np.stack([
                #             new_entries_rows,
                #             new_entries_col0s,
                #             new_entries_col1s,
                #             new_entries_f_index,
                #         ], axis=1).astype(int)
                    
                #     # TODO: uncomment
                #     # f_reading_table = np.vstack([outside_rows, new_reading_table_entries])
                #     # f_coord_table = np.vstack([outside_coords, new_coord_entries])
                #     f_reading_table = outside_rows
                #     f_coord_table = outside_coords

                # else:
                #     f_reading_table = outside_rows
                #     f_coord_table = outside_coords
                inside_rows   = (rows >= n_r0) & (rows < n_r1)
                outside_mask  = (~inside_rows) | (col1s <= n_c0) | (col0s >= n_c1)

                outside_rows = f_reading_table[outside_mask]

                # merge: outside + any left/right pieces collected in new_entries
                parts = [outside_rows]
                for part in new_entries:
                    if part is not None and part.size:
                        # ensure int dtype and 2D shape
                        parts.append(part.astype(int, copy=False).reshape(-1, 4))

                f_reading_table = np.vstack(parts) if len(parts) > 1 else outside_rows

                # Debugging

                # step_reading_table = f_reading_table.copy()
                # # step_coord_table = f_coord_table.copy()
                # # add column indicating if it was outside_rows or new_entries
                # # was_outside = np.zeros(len(step_reading_table), dtype=bool)
                # # was_outside[:len(outside_rows)] = True

                # # sort new reading table by rows
                # # sort_idx = np.lexsort((new_reading_table[:, 3], new_reading_table[:, 0]))
                # step_sort_idx = np.argsort(step_reading_table[:, 0])
                # step_reading_table = step_reading_table[step_sort_idx]
                # # step_coord_table = step_coord_table[step_sort_idx]
                # # was_outside = was_outside[step_sort_idx]
                # rows, row_starts = np.unique(step_reading_table[:, 0], return_index=True)
                # # yss = step_coord_table[:, 0]
                # # yss = yss[row_starts]
                # yss = (raster.transform * (np.zeros_like(rows), rows + 0.5))[1]

                # row_start, row_end = int(np.floor(window.row_off)), int(
                #     np.ceil(window.row_off + window.height)
                # )
                # col_start, col_end = int(np.floor(window.col_off)), int(
                #     np.ceil(window.col_off + window.width)
                # )

                # from raptorstats.debugutils import ref_mask_rasterstats, compare_stats, plot_mask_comparison
                
                # for i, row in enumerate(rows):
                #     start = row_starts[i]
                #     end = row_starts[i + 1] if i + 1 < len(row_starts) else len(step_reading_table)
                #     reading_line = step_reading_table[start:end]

                #     min_col = np.min(reading_line[:, 1])
                #     max_col = np.max(reading_line[:, 2])
                #     reading_window = rio.windows.Window(
                #         col_off=min_col, row_off=row, width=max_col - min_col, height=1
                #     )

                #     curr_y = yss[i]
                #     # if np.allclose(curr_y, 3851869.0157146556):
                #     print(f"Processing row {row} (y={curr_y})")

                #     # Does not handle nodata
                #     data = raster.read(1, window=reading_window, masked=True)
                #     if data.shape[0] == 0:
                #         continue
                #     data = data[0]

                #     for j, col0, col1, f_index in reading_line:
                #         row_in_mask = row - row_start
                #         col0_in_mask = col0 - col_start
                #         col1_in_mask = col1 - col_start
                #         global_mask[row_in_mask, col0_in_mask:col1_in_mask] = f_index + 1
                #         # if row == 512 or row == 479:
                #         #     global_mask[row_in_mask, col0_in_mask:col1_in_mask] = -1
                #         # if not was_outside[i]:
                #         #     global_mask[row_in_mask, col0_in_mask:col1_in_mask] = -1
                #         # else:
                #         #     global_mask[row_in_mask, col0_in_mask:col1_in_mask] = -1
                # # global_mask = global_mask[:-1, :]  # remove the last row added for debugging

                
                # ref_mask = ref_mask_rasterstats(features, raster, window)
                # print(f"Node Level: {node.level}")
                # # plot_mask_comparison(global_mask, ref_mask, features, raster.transform, window=window, scanlines=yss, idx=self.idx, used_nodes=used_nodes[f_index])
                # global_mask = np.zeros((H, W), dtype=int)  # reset global mask for next node
                # print('done')
                # End Debugging
            new_reading_table.append(f_reading_table)
            # new_coord_table.append(f_coord_table)

        new_reading_table = np.vstack(new_reading_table)
        # new_coord_table = np.vstack(new_coord_table)
        # sort new reading table by rows
        # sort_idx = np.lexsort((new_reading_table[:, 3], new_reading_table[:, 0]))
        sort_idx = np.argsort(new_reading_table[:, 0])
        reading_table = new_reading_table[sort_idx]
        # coord_table = new_coord_table[sort_idx]
        
        self.results = process_reading_table(
            reading_table, features, raster, self.stats, partials=partials
        )

        # Debugging
        # rows, row_starts = np.unique(reading_table[:, 0], return_index=True)

        # row_start, row_end = int(np.floor(window.row_off)), int(
        #     np.ceil(window.row_off + window.height)
        # )
        # col_start, col_end = int(np.floor(window.col_off)), int(
        #     np.ceil(window.col_off + window.width)
        # )

        # from raptorstats.debugutils import ref_mask_rasterstats, compare_stats, plot_mask_comparison

        # diffs = compare_stats(self.results, self.raster.files[0], features, stats=self.stats, show_diff=True, precision=5)
        
        # # diffs = []
        # if len(diffs) == 0:
        #     print("No differences found!!!")
        #     return self.results
        # diff_indices = [d['feature'] for d in diffs]
        # diff_features = features.iloc[diff_indices]
        # f_index_to_diff_features = { f: i for i, f in enumerate(diff_indices) }

        # for i, row in enumerate(rows):
        #     start = row_starts[i]
        #     end = row_starts[i + 1] if i + 1 < len(row_starts) else len(reading_table)
        #     reading_line = reading_table[start:end]

        #     min_col = np.min(reading_line[:, 1])
        #     max_col = np.max(reading_line[:, 2])
        #     reading_window = rio.windows.Window(
        #         col_off=min_col, row_off=row, width=max_col - min_col, height=1
        #     )

        #     # Does not handle nodata
        #     data = raster.read(1, window=reading_window, masked=True)
        #     if data.shape[0] == 0:
        #         continue
        #     data = data[0]

        #     for j, col0, col1, f_index in reading_line:
        #         row_in_mask = row - row_start
        #         col0_in_mask = col0 - col_start
        #         col1_in_mask = col1 - col_start
        #         if len(diff_indices) > 0:
        #             if f_index in diff_indices:
        #                 global_mask[row_in_mask, col0_in_mask:col1_in_mask] = f_index_to_diff_features[f_index] + 1
        #         else:
        #             global_mask[row_in_mask, col0_in_mask:col1_in_mask] = f_index + 1

        # global_mask = global_mask[:-1, :]  # remove the last row added for debugging

        
        # ref_mask = ref_mask_rasterstats(diff_features if len(diff_indices) > 0 else features, raster, window)
        # diff_used_nodes = []
        # for di in diff_indices:
        #     diff_used_nodes.append(used_nodes[di])
        # diff_used_nodes = [n for nlist in diff_used_nodes for n in nlist]
        # used_nodes= [ n for nlist in used_nodes for n in nlist ]
        # # used_nodes = []
        # print(diffs)
        # plot_mask_comparison(global_mask, ref_mask, diff_features if len(diff_indices) > 0 else features, raster.transform, window=window, scanlines=inter_ys, idx=self.idx, used_nodes=diff_used_nodes if len(diff_indices) >0 else used_nodes)
        # print('done')

        # End Debugging code

        return self.results


# def test_plot(boxes, aggregates, parent_ids, ids):

#     nx, ny = 8, 8
#     # normalize boxes to fit in the figure
#     x_min = min(boxes[:, [0, 2]].flatten())
#     x_max = max(boxes[:, [0, 2]].flatten())
#     y_min = min(boxes[:, [1, 3]].flatten())
#     y_max = max(boxes[:, [1, 3]].flatten())

#     boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_min) / (x_max - x_min) * nx
#     boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_min) / (y_max - y_min) * ny

#     fig, ax = plt.subplots(figsize=(nx, ny))

#     for i, (x0, y0, x1, y1) in enumerate(boxes):
#         rect = patches.Rectangle(
#             (x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="black", facecolor="none"
#         )
#         ax.add_patch(rect)

#         # Label the box with its index at the center
#         cx = (x0 + x1) / 2
#         cy = (y0 + y1) / 2
#         tt = aggregates[i]["count"]
#         pp = parent_ids[i]

#         color = "black"
#         if int(pp) in [1330, 1220, 1221, 1224]:
#             color = "red"

#         ii = ids[i]
#         ttt = str(ii) + " p:" + str(pp)
#         ax.text(cx, cy, ttt, fontsize=8, ha="center", va="center", color=color)

#     ax.set_xlim(min(boxes[:, 0]), max(boxes[:, 2]))
#     ax.set_ylim(min(boxes[:, 1]), max(boxes[:, 3]))
#     ax.set_aspect("equal")
#     ax.invert_yaxis()  # Optional: to mimic raster image top-down layout
#     plt.grid(True)
#     plt.show()
#     plt.savefig("test_plot.png")


# def plot_masks(raster, geom, my_mask, window):

#     correct_mask = Masking().mask(geom, raster, window)
#     # plot 3 things:
#     # my_mask
#     # correct_mask
#     # difference between the two masks
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     ax[0].imshow(my_mask, cmap="gray")
#     ax[0].set_title("My Mask")
#     ax[1].imshow(correct_mask, cmap="gray")
#     ax[1].set_title("Correct Mask")
#     ax[2].imshow(my_mask.astype(int) - correct_mask.astype(int), cmap="gray")
#     ax[2].set_title("Difference")
#     plt.show()
#     print("stop")
#     # add colorbar to the first two plots

    # plot difference between the two masks
    # mask - correct_mask
    # print('done')


# def test_raster():

#     # 10x10 raster with all 1s
#     array = np.ones((8, 8), dtype=np.uint8)
#     transform = rio.transform.from_origin(0, 8, 1, 1)  # (x_min, y_max, x_res, y_res)
#     from constants import RASTER_DATA_PATH

#     # Create the GeoTIFF file
#     with rio.open(
#         f"{RASTER_DATA_PATH}/test.tif",
#         "w",
#         driver="GTiff",
#         height=array.shape[0],
#         width=array.shape[1],
#         count=1,
#         dtype=array.dtype,
#         crs="EPSG:4326",  # WGS84
#         transform=transform,
#     ) as dst:
#         dst.write(array, 1)


# def test():

#     from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH

#     ag = AggQuadTree(max_depth=3)
#     vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
#     raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR.tif'

#     test_raster()
#     raster_layer_file = f'{RASTER_DATA_PATH}/test.tif'
#     ag(raster_layer_file, vector_layer_file, ['count', 'mean'])


# def test():
#     from constants import RASTER_DATA_PATH, VECTOR_DATA_PATH
#     from reference import reference_method

#     ag = AggQuadTree(max_depth=7)
#     vector_layer_file = f"{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp"
#     raster_layer_file = f"{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif"

#     # test_raster()
#     # raster_layer_file = f'{RASTER_DATA_PATH}/test.tif'
#     r1 = ag(raster_layer_file, vector_layer_file, ["count"])

#     rlow, rhigh = reference_method(raster_layer_file, vector_layer_file, ["count"])

#     for i in range(len(r1)):
#         r_c = r1[i]["count"]
#         r_l = rlow[i]["count"]
#         r_h = rhigh[i]["count"]
#         err1 = abs(r_c - r_l) / r_l
#         err2 = abs(r_c - r_h) / r_h
#         print(f"Error {i}: {err1:.2%}")
#         if err1 > 0.03:
#             print(f"r_c: {r_c}, r_l: {r_l}, r_h: {r_h}")

#     print("done")


# test()

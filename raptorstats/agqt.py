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

    def _compute_scanline_reading_table(
        self, features: gpd.GeoDataFrame, raster: rio.DatasetReader
    ):

        f_index, intersection_coords = build_intersection_table(features, raster)

        if intersection_coords.size == 0:
            reading_table = np.empty((0, 7), dtype=int)
            f_index_starts = np.array([], dtype=int)
            unique_f_indices = np.array([], dtype=int)
            self._reading_table = reading_table
            self.f_index_starts = f_index_starts
            self.unique_f_indices = unique_f_indices
            self.n_features = len(features.geometry)
            self.effective_n_features = 0
            return

        reading_table = build_reading_table(
            f_index, intersection_coords, raster, return_coordinates=False, sort_by_feature=True
        )

        self._reading_table = reading_table # row, col0, col1, f_index
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
                g = boxes[s:e]
                # union bounds
                out_boxes[k, 0] = np.min(g[:, 0])  # left
                out_boxes[k, 1] = np.min(g[:, 1])  # bottom
                out_boxes[k, 2] = np.max(g[:, 2])  # right
                out_boxes[k, 3] = np.max(g[:, 3])  # top
                out_ix[k] = pix[s]
                out_iy[k] = piy[s]
            return out_boxes, out_ix, out_iy

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

        # Build leaf level
        level = self.max_depth
        divisions = 2**level
        boxes, ix, iy = get_boxes_aligned_even(divisions, raster)

        previous_aggregates = None
        previous_parent_ids = None

        # Iterate from leaves up to root, coarsening
        for level in range(self.max_depth, -1, -1):
            divisions = 2**level

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
                parent_ids = (z_lvl // 4) + box_index  # parent ids live in the next chunk

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
                node = Node( # Node computes pixel_bounds from box and transform
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

            # Prepare parent level boxes by unioning children bounds
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
        f_index_starts = self.f_index_starts
        n_features = self.n_features
        n_eff_features = self.effective_n_features

        if reading_table.size == 0:
            # No intersections found, return empty stats
            return [self.stats.empty() for _ in range(n_features)]
        
        new_reading_table = []
        partials = [[] for _ in range(n_features)]

        # Debugging
        # H, W = raster.height, raster.width
        # window = rio.windows.Window(0, 0, W, H).round_offsets().round_lengths()  # for plotting/ref
        # global_mask = np.zeros((H, W), dtype=int)  # global mask for debugging
        # used_nodes = [[] for _ in range(n_features)]
        # End Debugging

        for f_index, geom in enumerate(features.geometry):

            if f_index not in self.unique_f_indices:
                # Skip features that do not have any intersections
                # self.unique_f_indices was computed from the scanline reading table
                # so if this f_index does not appear, it means there are n intersections
                continue

            # Get all intersecting nodes, sorted from root to leaves
            # window for geom
            window_bounds = geom.bounds
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

            f_index_start = f_index_starts[f_index]
            f_index_end = (
                f_index_starts[f_index + 1]
                if f_index + 1 < n_eff_features
                else len(reading_table)
            )
            f_reading_table = reading_table[f_index_start:f_index_end]

            if len(nodes) == 0:
                new_reading_table.append(f_reading_table)
                continue

            # Debugging
            # put node 1 in position 0
            # n1 = nodes[0]
            # n0 = nodes[0]
            # nodes[0] = nodes[1]
            # nodes[1] = n0
            # End Debugging

            for node in nodes:

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

                # Add the partials of this node
                partials[f_index].append(node.stats)

                rows = f_reading_table[:, 0]
                col0s = f_reading_table[:, 1]
                col1s = f_reading_table[:, 2]
                fully_inside = (rows >= n_r0) & (rows < n_r1) & (col0s >= n_c0) & (col1s <= n_c1)
                
                # Remove fully inside strips, as they are already accounted for
                # by the node
                f_reading_table = f_reading_table[~fully_inside]

                rows = f_reading_table[:, 0]
                col0s = f_reading_table[:, 1]
                col1s = f_reading_table[:, 2]

                f_idxs = f_reading_table[:, 3]

                new_entries = []

                # Left cut: segments or strips crossing the node's left edge
                # includes both the strips only crossing the left edge
                # and segments crossing both the left and right edges
                Lmask = (rows >= n_r0) & (rows < n_r1) & (col0s < n_c0) & (col1s > n_c0)

                if np.any(Lmask):
                        l_rows = rows[Lmask]
                        l_c0   = col0s[Lmask]
                        l_c1   = np.minimum(col1s[Lmask], n_c0) # cut at left edge
                        keep   = l_c1 > l_c0 # drop zero-length
                        # "Cut" the left edge, creating a new shorter segment in the left part
                        # that ends at the node's left edge
                        if np.any(keep):
                            left_reading = np.stack(
                                [l_rows[keep], l_c0[keep], l_c1[keep], f_idxs[Lmask][keep]],
                                axis=1
                            ).astype(int)
                            new_entries.append(left_reading)

                # Right cut: segments crossing the node's right edge
                # includes both the strips only crossing the right edge
                # and segments crossing both the left and right edges
                Rmask = (rows >= n_r0) & (rows < n_r1) & (col1s > n_c1) & (col0s < n_c1)

                if np.any(Rmask):
                    r_rows = rows[Rmask]
                    r_c0   = np.maximum(col0s[Rmask], n_c1)  # start at node's right edge
                    r_c1   = col1s[Rmask]
                    r_f    = f_idxs[Rmask]
                    # "Cut" the right edge, creating a new shorter segment in the right part
                    # that starts at the node's right edge
                    keep = r_c1 > r_c0  # drop zero-length pieces
                    if np.any(keep):
                        right_reading = np.stack(
                            [r_rows[keep], r_c0[keep], r_c1[keep], r_f[keep]],
                            axis=1
                        ).astype(int)
                        new_entries.append(right_reading)

                # Keep everything that is outside the node as is
                # outside means up and down the node, or completely
                # starting and ending to the left or right of the node
                inside_rows   = (rows >= n_r0) & (rows < n_r1)
                outside_mask  = (~inside_rows) | (col1s <= n_c0) | (col0s >= n_c1)
                outside_rows = f_reading_table[outside_mask]

                # merge: outside + any left/right pieces
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

        # Sort the lines by row so we can process them using scanline
        new_reading_table = np.vstack(new_reading_table)
        sort_idx = np.argsort(new_reading_table[:, 0])
        reading_table = new_reading_table[sort_idx]
        
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

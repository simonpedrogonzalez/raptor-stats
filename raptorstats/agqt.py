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
            return

        reading_table, coord_table = build_reading_table(
            f_index, intersection_coords, raster, return_coordinates=True, sort_by_feature=True
        )

        self._reading_table = reading_table # row, col0, col1, f_index
        self._coord_table = coord_table # y, x0, x1
        unique_f_indices, f_index_starts = np.unique(reading_table[:, 3], return_index=True)
        self.f_index_starts = f_index_starts
        self.unique_f_indices = unique_f_indices
        self.n_features = len(unique_f_indices)

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

        def get_boxes_aligned_any(divisions: int, raster: rio.DatasetReader):
            """
            Make a divisions x divisions grid of pixel-aligned tiles that together
            cover the whole raster. Tile widths/heights differ by at most 1 pixel.
            Returns (boxes, ix, iy) where boxes are (left, bottom, right, top).
            """
            from rasterio.windows import Window
            from rasterio.windows import bounds as win_bounds

            if divisions <= 0:
                raise ValueError("divisions must be > 0")

            W, H = raster.width, raster.height
            w_base, w_extra = divmod(W, divisions)  # first w_extra columns are w_base+1 px wide
            h_base, h_extra = divmod(H, divisions)  # first h_extra rows are h_base+1 px tall

            # per-column widths and per-row heights (pixel counts)
            widths  = np.concatenate([np.full(w_extra, w_base + 1, int),
                                    np.full(divisions - w_extra, w_base, int)])
            heights = np.concatenate([np.full(h_extra, h_base + 1, int),
                                    np.full(divisions - h_extra, h_base, int)])

            # integer pixel offsets
            col_offs = np.zeros(divisions, dtype=int)
            row_offs = np.zeros(divisions, dtype=int)
            col_offs[1:] = np.cumsum(widths)[:-1]
            row_offs[1:] = np.cumsum(heights)[:-1]

            boxes = []
            ix = []
            iy = []
            for iy_ in range(divisions):
                for ix_ in range(divisions):
                    win = Window(col_offs[ix_], row_offs[iy_], widths[ix_], heights[iy_])
                    # bounds: (left, bottom, right, top), aligned to pixel edges
                    boxes.append(win_bounds(win, raster.transform))
                    ix.append(ix_)
                    iy.append(iy_)

            return np.array(boxes, dtype=float), np.array(ix, dtype=int), np.array(iy, dtype=int)

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
        transform = raster.transform

        def xy_to_rowcol(xs, ys):
            # pixel in which the point is located
            cols, rows = (~transform) * (xs, ys)
            return np.floor(rows).astype(int), np.floor(cols).astype(int)

        new_reading_table = []
        new_coord_table = []
        partials = [[] for _ in range(n_features)]

        for f_index, geom in enumerate(features.geometry):

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
                if f_index + 1 < n_features
                else len(reading_table)
            )
            f_reading_table = reading_table[f_index_start:f_index_end]
            f_coord_table = coord_table[f_index_start:f_index_end]

            if len(nodes) == 0:
                new_reading_table.append(f_reading_table)
                new_coord_table.append(f_coord_table)
                continue

            # if f_index == 25:
            #     nnodes = [n for n in nodes if n.is_contained_in_geom(geom)]
            #     bbx = [n.box for n in nnodes]
            #     iids = [n.id for n in nnodes]
            #     aags = []
            #     piids = [n.parent_id for n in nnodes]
            #     # test_plot(bbx, aags, piids, iids)
            #     print(f"Feature index: {f_index}")

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

                ys = f_coord_table[:, 0]
                x0s = f_coord_table[:, 1]
                x1s = f_coord_table[:, 2]

                inside_y = (ys >= n_y0) & (ys <= n_y1)
                inside_x = (x0s >= n_x0) & (x1s <= n_x1)
                fully_inside = inside_y & inside_x

                f_reading_table = f_reading_table[~fully_inside]
                f_coord_table = f_coord_table[~fully_inside]

                ys = f_coord_table[:, 0]
                x0s = f_coord_table[:, 1]
                x1s = f_coord_table[:, 2]
                rows = f_reading_table[:, 0]
                
                f_idxs = f_reading_table[:, 3]
                # y, x0, x1, rows, col0s, col1s, f_index

                inside_y = (ys >= n_y0) & (ys <= n_y1)
                overlaps_left = (x0s < n_x0) & (x1s > n_x0)  # crosses left edge
                overlaps_right = (x1s > n_x1) & (x0s < n_x1)  # crosses right edge
                overlaps_both = (x0s < n_x0) & (x1s > n_x1)  # crosses both sides
                horizontally_intersect = (x0s < n_x1) & (x1s > n_x0)

                new_entries = []

                cut_left = (
                    horizontally_intersect & (overlaps_left | overlaps_both) & inside_y
                )
                n_cut_left = np.sum(cut_left)
                if n_cut_left > 0:
                    new_entry_left = np.stack(
                        [
                            ys[cut_left], # y
                            x0s[cut_left],  # new x0 is the left border
                            np.ones(n_cut_left) * n_x0,
                            rows[cut_left],
                            -1 * np.ones(n_cut_left),  # col0 placeholder
                            -1 * np.ones(n_cut_left),  # col1 placeholder
                            f_idxs[cut_left],
                        ],
                        axis=1,
                    )
                    new_entries.append(new_entry_left)

                cut_right = (
                    horizontally_intersect & (overlaps_right | overlaps_both) & inside_y
                )
                n_cut_right = np.sum(cut_right)
                if n_cut_right > 0:
                    new_entry_right = np.stack(
                        [
                            ys[cut_right],
                            np.ones(n_cut_right) * n_x1,  # new x0 is right border
                            x1s[cut_right],
                            rows[cut_right],
                            -1 * np.ones(n_cut_right),  # col0 placeholder
                            -1 * np.ones(n_cut_right),  # col1 placeholder
                            f_idxs[cut_right],
                        ],
                        axis=1,
                    )
                    new_entries.append(new_entry_right)

                completely_outside = ~inside_y | ~horizontally_intersect
                outside_rows = f_reading_table[completely_outside]
                outside_coords = f_coord_table[completely_outside]

                if new_entries:
                    
                    new_entries = np.vstack(new_entries)
                    new_entries_ys = new_entries[:, 0]
                    new_entries_x0s = new_entries[:, 1]
                    new_entries_x1s = new_entries[:, 2]
                    new_entries_f_index = new_entries[:, 6]
                    new_entries_rows = new_entries[:, 3]

                    new_coord_entries = np.stack([
                        new_entries_ys,
                        new_entries_x0s,
                        new_entries_x1s,
                    ], axis=1)
                    
                    _, new_entries_col0s = xy_to_rowcol(new_entries_x0s, new_entries_ys)
                    _, new_entries_col1s = xy_to_rowcol(new_entries_x1s, new_entries_ys)
                    
                    new_reading_table_entries = np.stack([
                            new_entries_rows,
                            new_entries_col0s,
                            new_entries_col1s,
                            new_entries_f_index,
                        ], axis=1).astype(int)
                    
                    f_reading_table = np.vstack([outside_rows, new_reading_table_entries])
                    f_coord_table = np.vstack([outside_coords, new_coord_entries])

                else:
                    f_reading_table = outside_rows
                    f_coord_table = outside_coords

                # if f_index == 25:
                #     print(25)

            new_reading_table.append(f_reading_table)
            new_coord_table.append(f_coord_table)

        new_reading_table = np.vstack(new_reading_table)
        new_coord_table = np.vstack(new_coord_table)
        # sort new reading table by rows
        # sort_idx = np.lexsort((new_reading_table[:, 0], new_reading_table[:, 3]))
        sort_idx = np.argsort(new_reading_table[:, 0])
        reading_table = new_reading_table[sort_idx]
        coord_table = new_coord_table[sort_idx]
        
        self.results = process_reading_table(
            reading_table, features, raster, self.stats, partials=partials
        )
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

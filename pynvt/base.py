import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colorbar
from matplotlib import colors
from skimage import exposure
from pynvt.afniio import AFNIIO
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


class PyNVTBase:

    # ==============================================================
    #  INIT
    # ==============================================================
    def __init__(self, *args, **kwargs):

        if len(args) != 4:
            raise Exception("[[[ERROR]]] [pyNVTBase.__init__]  Wrong Args")

        # args 1: underlay path
        # args 2: overlay path
        # args 3: plot result folder
        # args 4: plot result file name
        self.path_underlay = args[0]
        self.path_overlay = args[1]
        self.path_save = args[2]
        self.file_plot = args[3]

        # kwargs ------------------------
        # data_type   (default:UNDER_ONLY) = UNDER_ONLY, AFNI, ATLAS, ICA

        # AFNI
        # value_range (default:0- auto range)
        # p_value     (default:0.05)
        # a_value   (default:0.01)

        # ATLAS
        # path_label  (default:'')

        # choose_axis (default: y)         = x, y, z
        # space       (default: 0)
        # slice_range (default: (0,0)
        # slice_label (default: '') = both, slice_coordinate, slice_index
        # slice_coordinate       (default: 29)
        # slice_coordinate_value (default: -0.36)

        # fig_title   (default: '')
        # fig_n_column(default: 6)
        # transparent (default: 1.0)
        # dpi         (default: 300)
        # intensity_max  (default: 0.0)
        # intensity_min  (default: 0.0)

        # legend_bool    (default: False) = True, False
        # file_legend    (default: '')
        # leg_n_column   (default: 2)
        # leg_cell_width (default: 250)
        # leg_swatch_width (default: 50)
        # kwargs ------------------------

        # DataType
        self.data_type = kwargs["data_type"] if "data_type" in kwargs else "UNDER_ONLY"
        # Afni / ICA
        self.afni_brick_idx = kwargs["afni_brick_idx"] if "afni_brick_idx" in kwargs else 1
        self.value_range = kwargs["value_range"] if "value_range" in kwargs else 0
        self.p_value = kwargs["p_value"] if "p_value" in kwargs else 0.05
        self.a_value = kwargs["a_value"] if "a_value" in kwargs else 0.01
        # Atlas
        self.path_label = kwargs["path_label"] if "path_label" in kwargs else ''

        # Figure
        self.choose_axis = kwargs["choose_axis"] if "choose_axis" in kwargs else "y"
        self.space = kwargs["space"] if "space" in kwargs else 0
        self.slice_range = kwargs["slice_range"] if "slice_range" in kwargs else (0, 0)
        self.slice_label = kwargs["slice_label"] if "slice_label" in kwargs else ""
        self.slice_coordinate = kwargs["slice_coordinate"] if "slice_coordinate" in kwargs else 29
        self.slice_coordinate_value = kwargs["slice_coordinate_value"] if "slice_coordinate_value" in kwargs else -0.36

        # Design
        self.fig_title = kwargs["fig_title"] if "fig_title" in kwargs else ""
        self.fig_n_column = kwargs["fig_n_column"] if "fig_n_column" in kwargs else 6
        self.transparent = kwargs["transparent"] if "transparent" in kwargs else 1.0
        self.dpi = kwargs["dpi"] if "dpi" in kwargs else 300
        self.intensity_max = kwargs["intensity_max"] if "intensity_max" in kwargs else 0.0
        self.intensity_min = kwargs["intensity_min"] if "intensity_min" in kwargs else 0.0
        self.cmp_name = kwargs["cmp_name"] if "cmp_name" in kwargs else 'red_blue_default'
        self.bar_orientation = kwargs["bar_orientation"] if "bar_orientation" in kwargs else ''

        # Legend
        self.legend_bool = kwargs["legend_bool"] if "legend_bool" in kwargs else False
        self.file_legend = kwargs["file_legend"] if "file_legend" in kwargs else ''
        self.leg_n_column = kwargs["leg_n_column"] if "leg_n_column" in kwargs else 2
        self.leg_cell_width = kwargs["leg_cell_width"] if "leg_cell_width" in kwargs else 250
        self.leg_swatch_width = kwargs["leg_swatch_width"] if "leg_swatch_width" in kwargs else 50

        # ----- SET Values
        self.fig_n_row = 0
        self.selected_slices = []
        self.arr_resolution = []
        self.s_form_code = 0
        self.orientation = []
        self.data_underlay = []
        self.data_overlay = []
        self.arr_shape = []
        self.aspect_ratio = 0
        self.thickness = 0
        self.num_slices = 0
        self.fig_size = []
        self.max_value = 0
        self.min_value = 0
        self.color_map = None

    # -------------------------------------------------------
    #  Set Color map
    # -------------------------------------------------------
    def set_color_map(self):

        if self.cmp_name == 'red_blue_default':

            dict_color = {'red': ((0.0, 0.0, 0.0),
                                  (0.5, 0.0, 0.0),
                                  (0.5000000001, 0.8, 0.8),
                                  (1.0, 1.0, 1.0)),
                          'green': ((0.0, 1.0, 1.0),
                                    (0.40, 0.0, 0.0),
                                    (0.5, 0.0, 0.0),
                                    (0.60, 0.0, 0.0),
                                    (1.0, 1.0, 1.0)),
                          'blue': ((0.0, 1.0, 1.0),
                                   (0.4999999999, 1.0, 0.5),
                                   (0.5, 0.0, 0.0),
                                   (1.0, 0.0, 0.0))
                          }
            self.color_map = LinearSegmentedColormap(self.cmp_name, segmentdata=dict_color, N=256)
        else:
            self.color_map = cm.get_cmap(self.cmp_name, 256)

    # -------------------------------------------------------
    # Load Overlay Data Set
    # -------------------------------------------------------
    def load_overlay(self):

        if self.data_type == 'UNDER_ONLY':
            self.data_overlay = None
        else:

            if not os.path.isfile(self.path_overlay):
                raise Exception(
                    '[[[ERROR]]] [load_overlay] File Does not Exist [file path={}]'.format(self.path_overlay))

            try:
                # Load Overlay Image
                if self.data_type == 'AFNI':
                    self.data_overlay = nib.load(self.path_overlay).get_data()[:, :, :, 0, self.afni_brick_idx]
                elif self.data_type == 'ICA':
                    self.data_overlay = nib.load(self.path_overlay).get_data()[:, :, :, 0]
                elif self.data_type == 'ATLAS':
                    self.data_overlay = nib.load(self.path_overlay).get_data()[:, :, :]
                    # Make transparent
                    self.data_overlay = self.data_overlay.astype(float)
                    self.data_overlay[self.data_overlay == 0.] = np.nan
                else:
                    self.data_overlay = nib.load(self.path_overlay).get_data()[:, :, :]
            except Exception as e:
                print("[[[ERROR]]] [load_overlay] Fail to load the data [{}]".format(self.path_overlay))
                print(e)

    # -------------------------------------------------------
    #  Load AFNI Overlay
    # -------------------------------------------------------
    def load_afni_overlay(self):
        afni = AFNIIO(self.path_overlay, thr_p=self.p_value, thr_a=self.a_value)

        return afni.threshold_overlay(self.data_overlay, self.afni_brick_idx)

    # -------------------------------------------------------
    # Flip Data
    # -------------------------------------------------------
    def processing_data_flip(self, data, bool_swap):

        if data is None:
            print("[processing_data_flip] is None.")
            return data

        if self.s_form_code == 3:
            print("[processing_data_flip] Swap Axis ---")
            # Swap Axis
            print(data.shape)
            data = np.swapaxes(data, 1, 2)
            print(data.shape)

            # Swap resolution, orientation
            if bool_swap:
                self.arr_resolution[1], self.arr_resolution[2] = self.arr_resolution[2], self.arr_resolution[1]
                self.orientation[1], self.orientation[2] = self.orientation[2], self.orientation[1]

        if self.orientation[0] == 'R':
            data = nib.orientations.flip_axis(data, axis=0)
        if self.orientation[1] == 'A':
            data = nib.orientations.flip_axis(data, axis=1)
        if self.orientation[2] == 'S':
            data = nib.orientations.flip_axis(data, axis=2)

        return data

    # -------------------------------------------------------
    #  Calculate ratio
    # -------------------------------------------------------
    def get_aspect_ratio(self):

        if self.choose_axis == 'x':
            aspect_ratio = abs(float(self.arr_resolution[2]) / float(self.arr_resolution[1]))
        elif self.choose_axis == 'y':
            aspect_ratio = abs(float(self.arr_resolution[2]) / float(self.arr_resolution[0]))
        elif self.choose_axis == 'z':
            aspect_ratio = abs(float(self.arr_resolution[1]) / float(self.arr_resolution[0]))
        else:
            aspect_ratio = 1

        self.aspect_ratio = round(aspect_ratio, 1)

    # -------------------------------------------------------
    #   Thickness
    # -------------------------------------------------------
    def get_thickness(self):

        if self.choose_axis == 'x':
            self.thickness = abs(round(self.arr_resolution[0], 2))
        elif self.choose_axis == 'y':
            self.thickness = abs(round(self.arr_resolution[1], 2))
        elif self.choose_axis == 'z':
            self.thickness = abs(round(self.arr_resolution[2], 2))
        else:
            self.thickness = 0

        print("[get_thickness] thickness ==[{}] axis=[{}]".format(self.thickness, self.choose_axis))

    # -------------------------------------------------------
    #  Select slices using space parameter
    # -------------------------------------------------------
    def set_selected_slices(self):

        x, y, z = self.arr_shape
        spacing = self.space + 1

        if self.choose_axis == 'x':
            num_slices = x // spacing
            last_slice = x - spacing
            max_slice = x
        elif self.choose_axis == 'y':
            num_slices = y // spacing
            last_slice = y - spacing
            max_slice = y
        elif self.choose_axis == 'z':
            num_slices = z // spacing
            last_slice = z - spacing
            max_slice = z
        else:
            num_slices = 0
            last_slice = 0
            max_slice = 0

        start_slice = self.slice_range[0]
        end_slice = self.slice_range[1]
        print("[set_selected_slices] Range of selected slices: {}-{}".format(start_slice, end_slice))

        if start_slice <= 0 and end_slice <= 0:
            selected_slices = map(int, list(np.linspace(0, last_slice, num_slices)))
        else:
            if start_slice >= end_slice:
                raise Exception("[[[ERROR]]] [set_selected_slices] Wrong Slice Range")
            elif end_slice > max_slice:
                raise Exception("[[[ERROR]]] [set_selected_slices] Slice Range should be less than {}."
                                .format(max_slice))
            else:
                start_slice = start_slice - 1
                end_slice = end_slice
                num_slices = (end_slice - start_slice) // spacing

                if num_slices == 0:
                    raise Exception(
                        "[[[ERROR]]] [set_selected_slices] Space:{}-less than number of slices".format(self.space))
                else:
                    selected_slices = map(int, list(np.linspace(start_slice, end_slice - spacing, num_slices)))

        selected_slices = list(selected_slices)

        if self.choose_axis == 'x':
            data_under_selected = self.data_underlay[selected_slices, :, :]
            if self.data_overlay is None:
                data_over_selected = None
            else:
                data_over_selected = self.data_overlay[selected_slices, :, :]
            # Add total slice number at index 0
            selected_slices.insert(0, x)
        elif self.choose_axis == 'y':
            data_under_selected = self.data_underlay[:, selected_slices, :]
            if self.data_overlay is None:
                data_over_selected = None
            else:
                data_over_selected = self.data_overlay[:, selected_slices, :]
            # Add total slice number at index 0
            selected_slices.insert(0, y)
        elif self.choose_axis == 'z':
            data_under_selected = self.data_underlay[:, :, selected_slices]
            if self.data_overlay is None:
                data_over_selected = None
            else:
                data_over_selected = self.data_overlay[:, :, selected_slices]
            # Add total slice number at index 0
            selected_slices.insert(0, z)
        else:
            data_under_selected = self.data_underlay
            data_over_selected = self.data_overlay
            # Add total slice number at index 0
            selected_slices.insert(0, 0)

        print("[set_selected_slices] Number of selected slices: {}".format(num_slices))

        self.num_slices = num_slices
        self.data_underlay = data_under_selected
        self.data_overlay = data_over_selected
        self.selected_slices = selected_slices

    # -------------------------------------------------------
    #  Set layout number of rows and columns
    # -------------------------------------------------------
    def set_figure_num_row(self):

        self.fig_n_row = self.num_slices / self.fig_n_column
        rest = self.num_slices % self.fig_n_column
        if rest > 0:
            self.fig_n_row = self.fig_n_row + 1

        self.fig_n_row = int(self.fig_n_row)
        print("[set_figure_num_row] Row: {}, Column: {}".format(self.fig_n_row, self.fig_n_column))

    # -------------------------------------------------------
    #  Set Figure Size
    # -------------------------------------------------------
    def set_figure_size(self):

        # ratio = float(z) / x * aspect_ratio
        if self.choose_axis == 'x':
            width = abs(self.arr_resolution[1]) * self.arr_shape[1] * self.fig_n_column
            height = abs(self.arr_resolution[2]) * self.arr_shape[2] * self.fig_n_row
        elif self.choose_axis == 'y':
            width = abs(self.arr_resolution[0]) * self.arr_shape[0] * self.fig_n_column
            height = abs(self.arr_resolution[2]) * self.arr_shape[2] * self.fig_n_row
        elif self.choose_axis == 'z':
            width = abs(self.arr_resolution[0]) * self.arr_shape[0] * self.fig_n_column
            height = abs(self.arr_resolution[1]) * self.arr_shape[1] * self.fig_n_row
        else:
            width = 1
            height = 1

        print("[set_figure_size] Each slide: width={},height={}".format(width, height))

        if width > height:
            set_width = 10
            ratio = height / width
            self.fig_size = [set_width, set_width * ratio]
            print("[set_figure_size] set_width={},ratio={}".format(set_width, ratio))
        else:
            set_height = 10
            ratio = width / height
            self.fig_size = [set_height * ratio, set_height]
            print("[set_figure_size] set_height={},ratio={}".format(set_height, ratio))

        if self.slice_label != '':
            padding = 0.1 * self.fig_n_row
            self.fig_size = (self.fig_size[0], self.fig_size[1] + padding)

        print("[set_figure_size] Figure Size [width, height]: {}".format(self.fig_size))

    # -------------------------------------------------------
    # Processing intensity of data
    # -------------------------------------------------------
    def processing_data_intensity(self):

        if self.intensity_max == 0 and self.intensity_min == 0:

            p2 = np.percentile(self.data_underlay, 2)
            p98 = np.percentile(self.data_underlay, 98)
            print("[processing_data_intensity] p2: {}, p98: {}".format(p2, p98))

            # Return image after stretching or shrinking its intensity levels.
            self.data_underlay = exposure.rescale_intensity(self.data_underlay, in_range=(p2, p98))

        else:

            val_intensity_max = self.data_underlay.max() * self.intensity_max
            val_intensity_min = self.data_underlay.min() * self.intensity_min
            print("[processing_data_intensity] max: {}, min: {}".format(val_intensity_max, val_intensity_min))

            # Return image after stretching or shrinking its intensity levels.
            self.data_underlay = exposure.rescale_intensity(self.data_underlay,
                                                            in_range=(val_intensity_min, val_intensity_max))

    # -------------------------------------------------------
    # Set Label
    # -------------------------------------------------------
    def set_label_dict(self):
        label = dict()
        pattern = r'^\s+(?P<idx>\d+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+' \
                  r'\s+\d+\s+\d+\s+\d+\s+"(?P<roi>.*)"$'

        with open(self.path_label, 'r') as file_label:
            for line in file_label:
                re_match = re.match(pattern, line)
                if re_match:
                    re_grp = re_match.groups()
                    idx = int(re_grp[0])
                    rgb = re_grp[1:4]
                    roi = re_grp[4]
                    rgb = np.array(list(map(float, rgb))) / 255
                    label[idx] = roi, rgb

        label.pop(0, None)

        return label

    # -------------------------------------------------------
    #  Get Color Map for ATLAS
    # -------------------------------------------------------
    def get_color_map_from_label(self):

        # Label
        label = self.set_label_dict()

        # ROIs
        number_of_rois = len(label.keys())
        colors_for_rois = [label[idx][1] for idx in sorted(label.keys())]

        # Generate a colormap index
        bounds = np.linspace(0, number_of_rois, number_of_rois)
        if number_of_rois > 0:
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=number_of_rois)
        else:
            norm = None

        # Generate a colormap object
        color_map = colors.ListedColormap(colors_for_rois, 'indexed')

        return color_map, norm

    # -------------------------------------------------------
    #  Slice coordinate
    # -------------------------------------------------------
    def get_label_slice_coordinate(self, slice_num):
        slice_num = slice_num + 1

        if slice_num <= self.slice_coordinate:
            slice_coordinate_number = (self.slice_coordinate - slice_num) * self.thickness + self.slice_coordinate_value
        else:
            slice_coordinate_number = self.slice_coordinate_value - (slice_num - self.slice_coordinate) * self.thickness

        slice_coordinate_number = round(slice_coordinate_number, 2)
        return str(slice_coordinate_number)

    # -------------------------------------------------------
    #  Slice index
    # -------------------------------------------------------
    def get_label_slice_index(self, slice_num, total_slice_num):
        slice_num_label = '[' + str(slice_num + 1) + "/" + str(total_slice_num) + "]"
        return slice_num_label

    # -------------------------------------------------------
    #  Set Max, Min Values of AFNI
    # -------------------------------------------------------
    def set_overlay_range(self):

        if self.value_range == 0:

            check_overlay = np.isnan(self.data_overlay)
            if False in check_overlay:
                max_value = abs(np.nanmax(self.data_overlay))
                min_value = abs(np.nanmin(self.data_overlay))
            else:
                max_value = 0
                min_value = 0

            if max_value > min_value:
                self.max_value = max_value
                self.min_value = -max_value
            else:
                self.max_value = min_value
                self.min_value = -min_value
        else:
            self.max_value = self.value_range
            self.min_value = -self.value_range

        print("[set_overlay_range] Max: {} Min: {}".format(self.max_value, self.min_value))

    # -------------------------------------------------------
    #  Save Figure
    # -------------------------------------------------------
    def save_figure(self, figure, file_nm):
        try:
            if not os.path.exists(self.path_save):
                os.mkdir(self.path_save)

            path_figure = os.path.join(self.path_save, file_nm)
            figure.savefig(path_figure, bbox_inches="tight", pad_inches=0)

            print("\n** [save_figure] {}\n".format(path_figure))

        except Exception as e:
            print("[[[ERROR]]] Fail saving figure! file_nm={}".format(file_nm))
            print(e)

    # -------------------------------------------------------
    #  Make label legend for Atlas
    # -------------------------------------------------------
    def make_label_legend(self):

        print("# make_label_legend")

        # Label
        label = self.set_label_dict()

        # Set variable
        cell_height = 22
        margin = 20
        dpi = 72
        font_size = 14

        n = len(label)
        n_cols = self.leg_n_column
        n_rows = n // n_cols + int(n % n_cols > 0)

        width = self.leg_cell_width * n_cols + margin
        height = cell_height * n_rows + margin * 3

        figure_legend, ax_leg = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        figure_legend.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

        ax_leg.set_xlim(0, self.leg_cell_width * n_cols)
        ax_leg.set_ylim(cell_height * (n_rows - 0.5), - cell_height / 2.)
        ax_leg.yaxis.set_visible(False)
        ax_leg.xaxis.set_visible(False)

        idx_row = 1
        for idx in label.keys():
            row = (idx_row-1) % n_rows
            col = (idx_row-1) // n_rows
            y = row * cell_height

            swatch_start_x = self.leg_cell_width * col
            swatch_end_x = self.leg_cell_width * col + self.leg_swatch_width
            text_pos_x = self.leg_cell_width * col + self.leg_swatch_width + 7

            ax_leg.text(text_pos_x, y, label[idx][0],
                        fontsize=font_size, horizontalalignment='left',
                        verticalalignment='center')
            ax_leg.hlines(y, swatch_start_x, swatch_end_x, color=label[idx][1], linewidth=18)

            idx_row = idx_row + 1

        plt.rcParams['savefig.facecolor'] = "white"

        return figure_legend

    # -------------------------------------------------------
    #  Make Bar Legend for AFNI/ICA
    # -------------------------------------------------------
    def make_bar_legend(self):

        # Find the min and max of all colors for use in setting the color scale.
        v_max = self.max_value
        v_min = self.min_value
        print("[make_bar_legend] value MIN:{}, MAX{}".format(v_min, v_max))

        norm = colors.Normalize(vmin=v_min, vmax=v_max)

        if self.bar_orientation == 'horizontal':
            figure_legend, ax_leg = plt.subplots(figsize=(6, 1), facecolor='white')
            figure_legend.subplots_adjust(bottom=0.5)
            cb = colorbar.ColorbarBase(ax_leg, cmap=self.color_map, norm=norm, orientation='horizontal')
        else:
            figure_legend, ax_leg = plt.subplots(figsize=(0.5, 5), facecolor='white')
            cb = colorbar.ColorbarBase(ax_leg, cmap=self.color_map, norm=norm)

        plt.rcParams['savefig.facecolor'] = "white"

        return figure_legend


################################################################################################
#   class PyNVT
################################################################################################
class PyNVT(PyNVTBase):

    # ==============================================================
    #  Load Data: underlay, overlay, parameters
    # ==============================================================
    def load_data(self):

        # ------ Underlay Data
        if not os.path.isfile(self.path_underlay):
            raise Exception("[[[ERROR]]]  [load_data] File Does not Exist [file path={}]".format(self.path_underlay))

        data_underlay = nib.load(self.path_underlay)
        self.data_underlay = data_underlay.get_data()
        if len(self.data_underlay.shape) > 3:
            self.data_underlay = self.data_underlay[:, :, :, 0]

        # ------ Overlay Data
        self.load_overlay()

        # ------ Other parameters
        # load affine
        affine = data_underlay.affine
        # load header
        header = data_underlay.header
        affine_header = affine[:3, :3]
        print("[load_data] AFFINE HEADER==\n{}".format(affine_header))

        self.arr_resolution = affine_header.sum(axis=0)
        print("[load_data] resolution == {}".format(self.arr_resolution))

        # sform
        self.s_form_code = header['sform_code']
        print("[load_data] s_form_code == [{}]".format(self.s_form_code))

        # Orientation
        self.orientation = nib.aff2axcodes(affine)
        self.orientation = np.asarray(self.orientation)
        print("[load_data] orientation == [{}]".format(self.orientation))

        # ------ flip Data
        self.data_underlay = self.processing_data_flip(self.data_underlay, True)
        self.data_overlay = self.processing_data_flip(self.data_overlay, False)

        # ------ Shape
        self.arr_shape = self.data_underlay.shape
        print("[load_data] arr_shape == [{}]".format(self.arr_shape))

        # ------ Aspect ratio, Thickness
        self.get_aspect_ratio()
        self.get_thickness()
        print("[load_data] Aspect ratio ==[{}] thickness=[{}]".format(self.aspect_ratio, self.thickness))

        # ------ Set Figure Variables
        # Selected slices
        self.set_selected_slices()
        # number of rows
        self.set_figure_num_row()
        # Set Figure width, height
        self.set_figure_size()

        # ------ Processing intensity of data
        self.processing_data_intensity()

        # ------ Processing Overlay Data
        if self.data_type == 'AFNI' or self.data_type == 'ICA':
            self.data_overlay = self.load_afni_overlay()
            self.set_overlay_range()
        elif self.data_type == 'ATLAS':
            # Make transparent
            self.data_overlay = self.data_overlay.astype(float)
            self.data_overlay[self.data_overlay == 0.] = np.nan

    # ==============================================================
    #  Make figure
    # ==============================================================
    def make_figure(self):

        print("[make_figure] Number of Slices: [{}]".format(self.num_slices))
        print("[make_figure] Slice coordinate:{} value:{}".format(self.slice_coordinate, self.slice_coordinate_value))

        # Set Colormap
        norm = None
        if self.data_type == 'AFNI' or self.data_type == 'ICA':
            self.set_color_map()

        elif self.data_type == 'ATLAS':
            self.color_map, norm = self.get_color_map_from_label()

        # Plot Figure
        figure, axes = plt.subplots(self.fig_n_row, self.fig_n_column, figsize=[self.fig_size[0], self.fig_size[1]],
                                    dpi=self.dpi)
        plt.rcParams['savefig.facecolor'] = "black"
        if self.slice_label != "":
            plt.subplots_adjust(wspace=.0, hspace=.5)
        else:
            plt.subplots_adjust(wspace=.0, hspace=.1)

        # For loop Each Axes
        for i, ax in enumerate(axes.flatten()):

            # Set Title with file name
            if i == 0 and self.fig_title != '':
                ax.set_title(self.fig_title, color="white", loc='left')

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_visible(False)

            if i < self.num_slices:

                try:
                    if i == 0:
                        if self.choose_axis == 'x':
                            data_empty = self.data_underlay[i, :, :].T
                        elif self.choose_axis == 'y':
                            data_empty = self.data_underlay[:, i, :].T
                        elif self.choose_axis == 'z':
                            data_empty = self.data_underlay[:, :, i].T
                        data_empty = np.zeros(data_empty.shape)

                    if self.choose_axis == 'x':
                        ax.imshow(self.data_underlay[i, :, :].T, cmap='gray')
                    elif self.choose_axis == 'y':
                        ax.imshow(self.data_underlay[:, i, :].T, cmap='gray')
                    elif self.choose_axis == 'z':
                        ax.imshow(self.data_underlay[:, :, i].T, cmap='gray')

                    if self.data_type != 'UNDER_ONLY':

                        if self.data_type == 'AFNI' or self.data_type == 'ICA':

                            if self.choose_axis == 'x':
                                ax.imshow(self.data_overlay[i, :, :].T, cmap=self.color_map,
                                          vmax=self.max_value, vmin=self.min_value)
                            elif self.choose_axis == 'y':
                                ax.imshow(self.data_overlay[:, i, :].T, cmap=self.color_map,
                                          vmax=self.max_value, vmin=self.min_value)
                            elif self.choose_axis == 'z':
                                ax.imshow(self.data_overlay[:, :, i].T, cmap=self.color_map,
                                          vmax=self.max_value, vmin=self.min_value)

                        elif self.data_type == 'ATLAS':

                            if self.choose_axis == 'x':
                                ax.imshow(self.data_overlay[i, :, :].T, interpolation='nearest', norm=norm,
                                          cmap=self.color_map, alpha=self.transparent)
                            elif self.choose_axis == 'y':
                                ax.imshow(self.data_overlay[:, i, :].T, interpolation='nearest', norm=norm,
                                          cmap=self.color_map, alpha=self.transparent)
                            elif self.choose_axis == 'z':
                                ax.imshow(self.data_overlay[:, :, i].T, interpolation='nearest', norm=norm,
                                          cmap=self.color_map, alpha=self.transparent)
                    # END if data_type != 'UNDER_ONLY':

                    ax.set_aspect(self.aspect_ratio)

                    # ax.spines['bottom'].set_color('blue')
                    # ax.spines['top'].set_color('blue')
                    # ax.spines['right'].set_color('pink')
                    # ax.spines['left'].set_color('pink')

                    if self.slice_label != '':
                        axis_label = ''
                        if self.slice_label == 'slice_coordinate':
                            axis_label = self.get_label_slice_coordinate(self.selected_slices[i + 1])
                        elif self.slice_label == 'slice_index':
                            axis_label = self.get_label_slice_index(self.selected_slices[i + 1],
                                                                    self.selected_slices[0])
                        elif self.slice_label == 'both':
                            axis_label = self.get_label_slice_coordinate(self.selected_slices[i + 1]) + " " + \
                                         self.get_label_slice_index(self.selected_slices[i + 1],
                                                                    self.selected_slices[0])

                        ax.set_xlabel(axis_label, color='orange', fontsize='xx-small', backgroundcolor='black',
                                      verticalalignment='top', linespacing=0)

                except Exception as e:
                    print(e)
                    pass

            else:
                ax.imshow(data_empty, cmap='gray')

        # Save Figure
        self.save_figure(figure, self.file_plot)

    # ==============================================================
    #  Make Legend
    # ==============================================================
    def make_legend(self):

        if self.path_save == '' or self.file_legend == '':
            raise Exception("[[[ERROR]]]  Wrong legend path.")
        else:

            # Make Legend
            if self.data_type == 'ATLAS':
                figure_legend = self.make_label_legend()
            elif self.data_type == 'AFNI' or self.data_type == 'ICA':
                figure_legend = self.make_bar_legend()
            else:
                figure_legend = None

            # Save Legend
            if figure_legend is not None:
                self.save_figure(figure_legend, self.file_legend)

    # ==============================================================
    #  Print Available brick in afni
    # ==============================================================
    def print_avail_brick_afni(self):
        afni = AFNIIO(self.path_overlay)
        afni.print_avail_brick()

import legacy_utils.contour_class as contour_class
import numpy as np


class single_cell(object):
    def __init__(self, img_num, obj_num):
        self.img_num = img_num
        self.obj_num = obj_num

    # features

    def set_cell_features(self, feaure_name, feature_value):
        self.cell_features = dict(zip(feaure_name, feature_value))

    def set_traj_label(self, traj_label):
        self.traj_label = traj_label

    def set_am_flag(self, am_flag):
        self.am_flag = am_flag

    def set_size_variation(self, variation_value):
        self.size_variation = variation_value
        self.rel_size_variation = variation_value / self.cell_features["Cell_AreaShape_Area"]

    def set_xy_variation(self, xy_variation):
        self.xy_variation = xy_variation

    # cell contour
    def set_cell_contour(self, cell_contour):
        self.cell_contour = cell_contour

    def set_pca_cord(self, pca_cord):
        self.pca_cord = pca_cord

    def set_pca_cord_variation(self, variation_value):
        self.pca_cord_variation = variation_value

    def set_contour_variation(self, variation_vector):
        self.contour_variation = variation_vector

    # def set_mean_pca_cord_variation(self,variation_value):
    #     self.mean_pca_cord_variation=variation_value
    # def set_embed_cord(self,embed_cord):
    #     self.embed_cord=embed_cord
    # def set_embed_cord_variation(self,variation_value):
    #     self.embed_cord_variation=variation_value

    # def set_mean_contour_variation(self,variation_vector):
    #     self.mean_contour_variation=variation_vector
    # def set_contour_variation_cord(self,contour_vari_cord):
    #     self.contour_variation_cord=contour_vari_cord

    # def set_scale_embed_cord(self,scale_embed_cord):
    #     self.scale_embed_cord=scale_embed_cord
    # def set_scale_pca_cord(self,scale_pca_cord):
    #     self.scale_pca_cord=scale_pca_cord

    # neighbor
    def set_neighbor_info(self, img_border_flag, nonfree_border_ratio, neighbor_list):
        self.img_border_flag = img_border_flag
        self.nonfree_border_ratio = nonfree_border_ratio
        self.neighbor_list = neighbor_list
        self.neighbor_count = len(neighbor_list)

    # nucleus and its contour
    def set_nuc_info(self, nuc_obj_num, nuc_area, nc_ratio, nuc_intensity):
        self.nuc_obj_num = nuc_obj_num
        self.nuc_area = nuc_area
        self.nuc_ratio = nc_ratio
        self.nuc_intensity = nuc_intensity

    def set_nuc_contour(self, nuc_contour, nuc_center_r, nuc_center_theta, nuc_cell_axis_angle):
        self.nuc_contour = nuc_contour
        self.nuc_center_r = nuc_center_r
        self.nuc_center_theta = nuc_center_theta
        self.nuc_cell_axis_angle = nuc_cell_axis_angle

    def set_nuc_pca_cord(self, nuc_pca_cord):
        self.nuc_pca_cord = nuc_pca_cord

    def set_nuc_pca_cord_variation(self, variation_value):
        self.nuc_pca_cord_variation = variation_value

    def set_nuc_size_variation(self, variation_value):
        self.nuc_size_variation = variation_value

    def set_nuc_contour_variation(self, variation_vector):
        self.nuc_contour_variation = variation_vector

    # def set_mean_nuc_pca_cord_variation(self,variation_value):
    #     self.mean_nuc_pca_cord_variation=variation_value
    # def set_nuc_embed_cord(self,nuc_embed_cord):
    #     self.nuc_embed_cord=nuc_embed_cord
    # def set_nuc_embed_cord_variation(self,variation_value):
    #     self.nuc_embed_cord_variation=variation_value


# ----for converting parent instance to child instance----
class ConverterMixin(object):
    @classmethod
    def convert_to_class(cls, obj):
        obj.__class__ = cls


class fluor_single_cell(ConverterMixin, single_cell):
    def __init__(self, img_num, obj_num):
        super(fluor_single_cell, self).__init__(img_num, obj_num)

    def set_fluor_features(self, fluor_name, feature_list, feature_values):
        exec("self." + fluor_name + "_feature_list=feature_list")
        exec("self." + fluor_name + "_feature_values=feature_values")

    def set_fluor_pca_cord(self, fluor_feature_name, fluor_pca_cord):
        exec("self." + fluor_feature_name + "_pca_cord=fluor_pca_cord")

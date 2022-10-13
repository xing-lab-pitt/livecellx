class single_cell_traj(object):
    def __init__(self, traj_seri, traj_contour):
        self.traj_seri = traj_seri
        self.traj_contour = traj_contour

    def set_traj_feature(self, traj_feature):
        self.traj_feature = traj_feature

    def set_traj_morph_pca_cord(self, traj_cord):
        self.traj_morph_pca_cord = traj_cord

    def set_traj_cord_variation(self, traj_cord_variation):
        self.traj_cord_variation = traj_cord_variation

    def set_traj_nuc_intensity(self, traj_nuc_intensity):
        self.traj_nuc_intensity = traj_nuc_intensity

    def set_traj_delay_vector(self, traj_delay_vector):
        self.traj_delay_vector = traj_delay_vector

    def set_traj_delay_cord(self, traj_delay_cord):
        self.traj_delay_cord = traj_delay_cord

    def set_traj_divide_points(self, divide_points):
        self.traj_divide_points = divide_points

    def set_traj_am_flag(self, traj_am_flag):
        self.traj_am_flag = traj_am_flag

    def set_traj_scale_cord(self, traj_scale_cord):
        self.traj_scale_cord = traj_scale_cord

    def set_traj_neighbor(self, traj_neighbor):
        self.traj_neighbor = traj_neighbor


# ----for converting parent instance to child instance----
class ConverterMixin(object):
    @classmethod
    def convert_to_class(cls, obj):
        obj.__class__ = cls


class fluor_single_cell_traj(ConverterMixin, single_cell_traj):
    def __init__(self, traj_seri, traj_contour):
        super(fluor_single_cell_traj, self).__init__(traj_seri, traj_contour)

    def set_traj_fluor_features(self, fluor_name, feature_list, traj_fluor_feature_values):
        exec("self." + fluor_name + "_feature_list=feature_list")
        exec("self.traj_" + fluor_name + "_feature_values=traj_fluor_feature_values")

    def set_traj_fluor_pca_cord(self, fluor_feature_name, traj_fluor_pca_cord):
        exec("self.traj_" + fluor_feature_name + "_pca_cord=traj_fluor_pca_cord")

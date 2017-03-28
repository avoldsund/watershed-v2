from osgeo import gdal  # For reading tiff files
import numpy as np


class Landscape:
    def __init__(self, ds):
        geo_transform = ds.GetGeoTransform()
        self.nx = ds.RasterXSize
        self.ny = ds.RasterYSize
        self.x_min = geo_transform[0]
        self.y_max = geo_transform[3]
        self.x_max = self.x_min + geo_transform[1] * (self.nx - 1)
        self.y_min = self.y_max + geo_transform[5] * (self.ny - 1)
        self.total_nodes = self.nx * self.ny
        self.heights = None

        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal'
            return
        self.step_size = step_size_x


def load_ds(filename):
    """
    Load the geotiff data set from the filename
    :param filename: Name of the .tiff file
    :return ds: Geotiff data
    """

    ds = gdal.Open(filename)

    if ds is None:
        print "Error retrieving data set."
        return

    return ds


def get_array_from_band(ds, band=1):
    """
    Get data from selected band and remove invalid data points
    :param ds: Geotiff data
    :param band: Different bands hold different information
    :return arr: The 2D-array holding the data
    """

    band = ds.GetRasterBand(band)
    arr = np.ma.masked_array(band.ReadAsArray())
    no_data_value = band.GetNoDataValue()

    if no_data_value:
        arr[arr == no_data_value] = np.ma.masked

    return arr


def get_landscape(file_name):
    """
    Returns a landscape object given a tiff file. This object contains all coordinates, downslope neighbors etc.
    :param file_name: File name of the .tiff data file
    :return landscape: Landscape object
    """

    data_set = load_ds(file_name)
    heights = get_array_from_band(data_set)
    landscape = Landscape(data_set)
    landscape.heights = heights

    return landscape


def get_landscape_tyrifjorden(file_name):
    """
    Return a modified landscape object because the Tyrifjorden area have NaN values for first column and last row
    :param file_name: Name of .tiff file
    :return landscape: The modified landscape
    """

    data_set = load_ds(file_name)
    heights = get_array_from_band(data_set)
    heights = heights[0:-1, 1:]

    landscape = Landscape(data_set)
    landscape.heights = heights
    landscape.nx -= 1
    landscape.ny -= 1
    landscape.x_min += landscape.step_size
    landscape.y_min += landscape.step_size
    landscape.total_nodes = landscape.nx * landscape.ny

    return landscape


def get_smallest_test_landscape_tyrifjorden(file_name):

    size = 500
    data_set = load_ds(file_name)
    heights = get_array_from_band(data_set)
    heights = heights[0:-1, 1:]
    heights = heights[0:size, 0:size]

    landscape = Landscape(data_set)
    landscape.heights = heights
    landscape.nx = size
    landscape.ny = size
    landscape.x_min += landscape.step_size
    landscape.y_min -= landscape.step_size * (landscape.nx - size)
    landscape.total_nodes = landscape.nx * landscape.ny

    return landscape


def get_smallest_test_landscape(file_name):

    size = 1000
    data_set = load_ds(file_name)
    heights = get_array_from_band(data_set)
    heights = heights[0:size, 0:size]

    landscape = Landscape(data_set)
    landscape.heights = heights
    landscape.nx = size
    landscape.ny = size
    landscape.x_min += landscape.step_size
    landscape.y_min -= landscape.step_size * (landscape.nx - size)
    landscape.total_nodes = landscape.nx * landscape.ny

    return landscape

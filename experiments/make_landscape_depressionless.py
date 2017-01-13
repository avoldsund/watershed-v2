from lib import util, load_data

"""
Alters the height in a landscape so that it becomes depressionless
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

landscape = load_data.get_landscape_tyrifjorden(file_name)
util.make_depressionless(landscape.heights, landscape.step_size)


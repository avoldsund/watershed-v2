from lib import util, load_data, river_analysis
import numpy as np

"""
Alters the height in a landscape so that it becomes depressionless
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

#%landscape = load_data.get_landscape_tyrifjorden(file_name)

#util.make_depressionless(landscape.heights, landscape.step_size)

heights = np.array([[4, 10, 10, 10, 10, 10, 10],
                    [10, 1, 8, 7, 7, 7, 10],
                    [10, 6, 8, 5, 5, 5, 10],
                    [10, 8, 8, 4, 3, 3, 10],
                    [10, 9, 9, 3, 3, 3, 10],
                    [10, 0, 1, 5, 5, 5, 10],
                    [10, 10, 10, 10, 10, 10, 10]])

util.make_depressionless(heights, 10, False)
print heights
#a, b, c = util.calculate_watersheds(heights, 7, 7, 10, False)
#print a
#print b
#print c
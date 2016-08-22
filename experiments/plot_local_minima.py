from lib import plot, load_data, util
import numpy as np
import matplotlib.pyplot as plt
import time

file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_landscape_tyrifjorden(file_name)

start = time.time()
util.fill_single_cell_depressions(landscape.heights, landscape.interior_nx, landscape.interior_ny)
end = time.time()
print end-start

plt.show()



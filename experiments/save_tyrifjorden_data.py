from lib import load_data, river_analysis
import cPickle as pickle

"""
Save all information from a landscape necessary to find the watershed information from an arbitrary outlet
"""

file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_landscape_tyrifjorden(file_name)

d4 = False
landscape, watersheds, steepest, flow_directions, spill_heights, traps, size_of_traps, expanded_conn_mat = \
    river_analysis.calculate_all_data(landscape, d4)

pickle.dump(landscape, open('landscapeTyrifjorden.pkl', 'wb'))
pickle.dump(watersheds, open('watershedsTyrifjorden.pkl', 'wb'))
pickle.dump(flow_directions, open('flowDirectionsTyrifjorden.pkl', 'wb'))
pickle.dump(expanded_conn_mat, open('connMatTyrifjorden.pkl', 'wb'))

pickle.dump(traps, open('trapsTyrifjorden.pkl', 'wb'))
pickle.dump(steepest, open('steepestTyrifjorden.pkl', 'wb'))
pickle.dump(spill_heights, open('spillHeightsTyrifjorden.pkl', 'wb'))
pickle.dump(size_of_traps, open('sizeOfTrapsTyrifjorden.pkl', 'wb'))

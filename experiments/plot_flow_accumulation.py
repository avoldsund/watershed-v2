from lib import plot
import cPickle as pickle

"""
Plot accumulated flow, either by loading it, or calculating it and saving
"""
saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/masters-thesis/'

# Save accumulated flow
# file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
# landscape = load_data.get_smallest_test_landscape_tyrifjorden(file_name)
# landscape = load_data.get_landscape_tyrifjorden(file_name)
# acc_flow = river_analysis.calculate_accumulated_flow(landscape, d4=False)
# pickle.dump(acc_flow, open('accFlowTyrifjorden.pkl', 'wb'))

# Load accumulated flow
acc_flow = pickle.load(open(saved_files + 'accFlowTyrifjorden.pkl', 'rb'))
plot.plot_accumulated_flow(acc_flow)

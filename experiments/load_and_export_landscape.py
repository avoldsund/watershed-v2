from lib import river_analysis
import scipy.io
import cPickle as pickle

"""
Export a watershed given an outlet
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/masters-thesis/'
file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'

# Load all necessary data
landscape = pickle.load(open(saved_files + 'landscapeTyrifjorden.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watershedsTyrifjorden.pkl', 'rb'))
flow_directions = pickle.load(open(saved_files + 'flowDirTyrifjorden.pkl', 'rb'))
conn_matrix = pickle.load(open(saved_files + 'connMatTyrifjorden.pkl', 'rb'))
traps = pickle.load(open(saved_files + 'trapsTyrifjorden.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestTyrifjorden.pkl', 'rb'))
spill_heights = pickle.load(open(saved_files + 'spillHeightsTyrifjorden.pkl', 'rb'))
size_of_traps = pickle.load(open(saved_files + 'sizeOfTrapsTyrifjorden.pkl', 'rb'))

# Get watershed of desired watershed outlet outlet_coords_r_c
step_size = 10
outlet_coords_r_c = (3269, 1041)
ws_of_node, trap_indices_in_ws = river_analysis.get_watershed_of_node(outlet_coords_r_c, conn_matrix, traps, landscape.ny, landscape.nx)

# Pre-process data
heights, ws, traps_in_ws, trap_heights, total_trap_cells, nr_of_traps, nr_of_cells_in_each_trap, flow_directions, spill_pairs = \
    river_analysis.pre_process_export_data(landscape, ws_of_node, traps, trap_indices_in_ws, spill_heights,
                                           flow_directions, steepest_spill_pairs)

# Save all landscape information
scipy.io.savemat('landscape.mat', dict(watershed=ws, outlet=outlet_coords_r_c, stepSize=step_size, heights=heights,
                                       traps=traps_in_ws, totalTrapCells=total_trap_cells, nrOfTraps=nr_of_traps,
                                       nrOfCellsInEachTrap=nr_of_cells_in_each_trap, trapHeights=trap_heights,
                                       spillPairs=steepest_spill_pairs, flowDirections=flow_directions))

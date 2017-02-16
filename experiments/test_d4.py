import numpy as np
from matplotlib import pyplot as plt
from lib import load_data, util, river_analysis, plot
import scipy.io

# heights = np.array([[10, 10, 10, 10, 10, 10, 10],
#                     [4, 1, 8, 7, 7, 7, 10],
#                     [10, 6, 8, 5, 5, 5, 10],
#                     [10, 8, 8, 4, 2, 4, 10],
#                     [10, 9, 9, 3, 3, 3, 10],
#                     [10, 0, 1, 5, 5, 5, 10],
#                     [10, 10, 10, 10, 10, 10, 10]])
# dim_y = 7
# dim_x = 7

# heights = np.array([[10, 10, 10, 10, 10, 10, 10, 10],
#                     [10, 4, 5, 4, 15, 3, 9, 2],
#                     [10, 10, 10, 10, 10, 10, 10, 10]])
# dim_x = 8
# dim_y = 3

heights = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 1, 1, 1, 2, 1, 1, 1, 3],
                    [3, 1, 0, 1, 2, 1, 0, 1, 3],
                    [3, 1, 1, 1, 2, 1, 1, 1, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3]])
dim_x = 9
dim_y = 5

step_size = 10
flow_directions = util.get_flow_direction_indices(heights, step_size, dim_y, dim_x, d4=True)
node_endpoints = util.get_node_endpoints(flow_directions)
local_watersheds = util.get_local_watersheds(node_endpoints)
local_minima = np.asarray(local_watersheds.keys())
combined_minima = util.combine_minima(local_minima, dim_y, dim_x, d4=True)
watersheds = util.combine_watersheds(local_watersheds, combined_minima)
print 'flow_directions: ', flow_directions
print 'node_endpoints: ', node_endpoints
print 'local_watersheds: ', local_watersheds
print 'local_minima: ', local_minima
print 'combined_minima: ', combined_minima
print 'watersheds: ', watersheds
boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, dim_x, dim_y, d4=True)
print 'possible spill pairs: ', util.get_possible_spill_pairs(heights, boundary_pairs)
watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, heights, d4=True)

print 'watersheds after combining: ', watersheds
print 'steepest_spill_pairs: ', steepest_spill_pairs

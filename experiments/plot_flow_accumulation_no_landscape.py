import numpy as np
from lib import river_analysis, plot

heights = np.array([[10, 10, 10, 10, 10, 10],
                                    [10, 9, 9, 9, 7, 10],
                                    [10, 9, 10, 9, 7, 10],
                                    [8, 10, 10, 10, 7, 10],
                                    [10, 4, 4, 4, 4.5, 10],
                                    [10, 4, 10, 10, 10, 10]])

outlet = (4, 2)
step_size = 10
ny, nx = np.shape(heights)

# flow_acc = river_analysis.calculate_accumulated_flow_no_landscape(heights, step_size, False)
# plot.plot_flow_acc_no_landscape(flow_acc)

ws_of_node, traps, trap_heights, trap_indices_in_ws, steepest_spill_pairs, flow_directions, heights = river_analysis.\
    calculate_watershed_of_node_no_landscape_input(heights, nx, ny, step_size, outlet, d4=False)

plot.plot_watershed_of_node(ws_of_node, ny, nx)

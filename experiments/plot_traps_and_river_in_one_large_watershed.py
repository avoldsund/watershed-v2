from lib import load_data, plot, util, analysis
import cPickle as pickle
import numpy as np
import networkx as nx

"""
Plot all rivers from the spill points
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

landscape = load_data.get_landscape_tyrifjorden(file_name)

watersheds = pickle.load(open(saved_files + 'watershedsThreshold0.pkl', 'rb'))
watersheds_threshold = pickle.load(open(saved_files + 'watershedsThreshold2500.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
flow_directions = pickle.load(open(saved_files + 'flowDirections.pkl', 'rb'))

mapping = util.map_nodes_to_watersheds(watersheds, landscape.ny, landscape.nx)
steepest_spill_pairs = list(steepest_spill_pairs)
order = np.argsort([mapping[s[0]] for s in steepest_spill_pairs])
steepest_spill_pairs = [steepest_spill_pairs[el] for el in order]
spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
all_traps, size_of_traps = util.get_all_traps(watersheds, landscape.heights, spill_heights)

# The large watershed to be plotted with all its small ones
merged_watersheds = [np.unique(mapping[ws]) for ws in watersheds_threshold]
focused_ws = merged_watersheds[0]  # The indices of the small watersheds in a large thresholded one

# Find river of watersheds
spill_pairs_between_ws = [(mapping[el[0]], mapping[el[1]]) for el in steepest_spill_pairs if mapping[el[0]] in focused_ws or mapping[el[1]] in focused_ws]
start_end = [el for el in spill_pairs_between_ws if el[0] not in focused_ws or el[1] not in focused_ws]
start = [el[0] for el in start_end if el[0] in focused_ws]
end = [el[1] for el in start_end if el[1] in focused_ws]
end_pair = [el for el in start_end if el[1] in focused_ws]

h1 = landscape.heights[util.map_1d_to_2d(end_pair[0][0], landscape.nx)]
h2 = landscape.heights[util.map_1d_to_2d(end_pair[0][1], landscape.nx)]
spill_height = max(h1, h2)

G = nx.Graph()
G.add_edges_from(spill_pairs_between_ws)
river_ws = nx.shortest_path(G, start[0], end[0])
traps_in_river = [all_traps[i] for i in range(len(all_traps)) if i in river_ws]
print 'hei'
# Plot all the traps in the river
# plot.plot_traps_in_river_2d(watersheds, traps_in_river, focused_ws, river_ws, landscape, 1)
rivers = analysis.get_rivers(watersheds, [watersheds_threshold[0]], steepest_spill_pairs, all_traps,
                             flow_directions, landscape.heights)
rivers = np.concatenate(rivers)
selected_watersheds = [watersheds[i] for i in range(len(watersheds)) if i in focused_ws]

plot.plot_traps_and_river_in_one_watershed_2d(traps_in_river, selected_watersheds, rivers, landscape, 1)

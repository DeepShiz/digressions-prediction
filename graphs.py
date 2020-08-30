import pandas as pd
import matplotlib.pyplot as plt

filename_breakout = "datasets/filtered_data_breakouts_refined.xlsx"
filename_goodcast = "datasets/filtered_data_goodcasts.xls"
graph_path = "Graphs/"

def plotter(x, y, graph_name):
    plt.plot(x, y, marker='o', linestyle='dashed', markersize=2)
    plt.show()
    #plt.savefig(graph_name, bbox_inches='tight')
    return 1

def get_x_y(data, min, max):
    level_vals = []
    for elem in data:
        level_vals += [elem[0]]

    time_instants = [i for i in range(min,max)]
    return level_vals, time_instants

def plot_breakout_last_100(data, graph_name="temp"):
    level_vals, time_instants = get_x_y(data, 1101,1201)
    plotter(time_instants, level_vals, graph_path + graph_name)
    return level_vals, time_instants

def plot_before_breakout_instant(data, graph_name="temp"):
    level_vals, time_instants = get_x_y(data, 1150,1173)
    plotter(time_instants, level_vals, graph_path + graph_name)
    return level_vals, time_instants

def plot_breakout(data, graph_name="temp"):
    level_vals, time_instants = get_x_y(data, 0,1167)
    plotter(time_instants, level_vals, graph_path + graph_name)
    return level_vals, time_instants

def plot_goodcast(data, graph_name="temp"):
    level_vals = []
    for elem in data:
        level_vals += [elem[1]]

    time_instants = [i for i in range(0,1201)]
    plotter(time_instants, level_vals, graph_path + graph_name)
    return level_vals, time_instants

if(__name__=='__main__'):
    data_breakout = pd.read_excel(filename_breakout)
    data_goodcast = pd.read_excel(filename_goodcast)

    data_breakout = pd.DataFrame(data_breakout)
    data_goodcast = pd.DataFrame(data_goodcast)

    data_refined_breakout = data_breakout.values.tolist()
    data_refined_goodcast = data_goodcast.values.tolist()

    data_blc = data_refined_breakout[1101:1201]
    data_bbi = data_refined_breakout[1150:1173]
    data_bf = data_refined_breakout[0:1167]
    data_gc = data_refined_goodcast[0:1201]

    level_vals_blc, time_instants_blc = plot_breakout_last_100(data_blc, "Breakout Last 100")
    level_vals_bbi, time_instants_bbi = plot_before_breakout_instant(data_bbi, "Breakout Instant")
    level_vals_gc, time_instants_gc = plot_goodcast(data_gc, "Goodcast")
    level_vals_bf, time_instants_bf = plot_breakout(data_bf, "Breakout")

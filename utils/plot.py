import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_corr(scores_1,
              scores_2, 
              outfile,
              x_label='',
              y_label='',
              jitter=False):

    res = stats.spearmanr(scores_1, scores_2)
    correlation = round(res.correlation, 3)
    pvalue = round(res.pvalue, 3)

    if jitter:
        plt.scatter(rand_jitter(scores_1), rand_jitter(scores_2), s=1, alpha=1)
    else:
        plt.scatter(scores_1, scores_2, s=1, alpha=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if pvalue < 0.05:
        pvalue = 'p<0.05'
    else:
        pvalue = f"p={pvalue}"
    print(pvalue)

    plt.title(f"Spearman={correlation} ({pvalue})")

    #plt.tick_params(labelleft=False, length=0)
    #plt.tick_params(labelbottom=False, length=0)
    #plt.yscale('log')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()

def read_file(path):
    with open(path) as f:
        data = json.load(f)
    data = np.array(data)
    return data 

if __name__ == "__main__":
    data_dir = '/home/cljiao/InContextDataValuation/outputs/gathered_scores'

    ft_same_task_file = f"{data_dir}/ft_same_all.json"
    ft_different_task_file = f"{data_dir}/ft_different_all.json"
    icp_same_task_file = f"{data_dir}/icp_same_all.json"
    icp_different_task_file = f"{data_dir}/icp_different_all.json"
    infl_same_task_file = f"{data_dir}/infl_same_all.json"
    infl_different_task_file = f"{data_dir}/infl_different_all.json"

    icp_same_task = read_file(icp_same_task_file)[0]
    ft_same_task = read_file(ft_same_task_file)[0]
    infl_same_task = read_file(infl_same_task_file)[0]

    icp_different_task = read_file(icp_different_task_file)[0]
    ft_different_task = read_file(ft_different_task_file)[0]
    infl_different_task = read_file(infl_different_task_file)[0]

    plots_dir = "/home/cljiao/InContextDataValuation/outputs/plots"

    plot_corr(icp_same_task, ft_same_task, f"{plots_dir}/icp_ft_same_task.png", x_label='ICP', y_label='FT')
    plot_corr(ft_same_task, infl_same_task, f"{plots_dir}/ft_infl_same_task.png", x_label='FT', y_label='Infl')
    plot_corr(icp_same_task, infl_same_task, f"{plots_dir}/icp_infl_same_task.png", x_label='ICP', y_label='Infl')

    plot_corr(icp_different_task, ft_different_task, f"{plots_dir}/icp_ft_different.png", x_label='ICP', y_label='FT')
    plot_corr(ft_different_task, infl_different_task, f"{plots_dir}/ft_infl_different.png", x_label='FT', y_label='Infl')
    plot_corr(icp_different_task, infl_different_task, f"{plots_dir}/icp_infl_different.png", x_label='ICP', y_label='Infl')
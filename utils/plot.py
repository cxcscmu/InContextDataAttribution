import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats


data_dir = '/home/cljiao/InContextDataValuation/outputs/gathered_scores_kmeans'
baseline_dir = '/home/cljiao/InContextDataValuation/outputs/gathered_scores_baseline_kmeans'
plots_dir = "/home/cljiao/InContextDataValuation/outputs/plots"

def get_avg_spearman(scores_1, scores_2):
    spearmans = []
    p_values = []
    for s1, s2 in zip(scores_1, scores_2):
        res = stats.spearmanr(s1, s2)
        correlation = round(res.correlation, 3)
        pvalue = round(res.pvalue, 3)
        spearmans.append(correlation)
        p_values.append(pvalue)
        print(correlation)
    
    return np.mean(spearmans), np.mean(p_values)

def scatter_plot(scores_1,
                 scores_2, 
                 outfile,
                 x_label='',
                 y_label='',
                 jitter=False):

    plt.scatter(scores_1, scores_2, s=1, alpha=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()

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
    #plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()


def read_file(path):
    with open(path) as f:
        data = json.load(f)
    data = np.array(data)
    return data 

def task_corr(task):
    icp_file = f"{data_dir}/icp_{task}_all.json"
    ft_scores = f"{data_dir}/ft_{task}_all.json"
    infl_file = f"{data_dir}/infl_{task}_all.json"

    icp_scores = read_file(icp_file)
    ft_scores = read_file(ft_scores)
    infl_scores = read_file(infl_file)

    print('Avg. ICP-Infl')
    print(get_avg_spearman(icp_scores, infl_scores))

    print('Avg. ICP-FT')
    print(get_avg_spearman(icp_scores, ft_scores))
    
    print('Avg. FT-Infl')
    print(get_avg_spearman(ft_scores, infl_scores))


def change_in_score(task, outdir=None):
    def repeat_baselines(baselines, n_prompts):
        baselines = np.expand_dims(baselines, axis=1)
        baselines = np.repeat(baselines, n_prompts, 1)
        return baselines

    icp_scores = np.array(read_file(f"{data_dir}/icp_{task}_all.json"))
    ft_scores =  np.array(read_file(f"{data_dir}/ft_{task}_all.json"))
    infl_scores =  np.array(read_file(f"{data_dir}/infl_{task}_all.json"))

    baseline_icp_scores = np.array(read_file(f"{baseline_dir}/icp_{task}.json"))
    baseline_ft_scores = np.array(read_file(f"{baseline_dir}/ft_{task}.json"))
    #baseline_ft_scores = np.squeeze(baseline_ft_scores, -1)

    print('icp_scores', icp_scores.shape)
    print('ft_scores', ft_scores.shape)
    print('infl_scores', infl_scores.shape)

    print('baseline_icp_scores', baseline_icp_scores.shape)
    print('baseline_ft_scores', baseline_ft_scores.shape)

    baseline_icp_scores = repeat_baselines(baseline_icp_scores, len(icp_scores[0]))
    baseline_ft_scores = repeat_baselines(baseline_ft_scores, len(ft_scores[0]))

    #plot_corr(infl_scores.flatten(), icp_scores.flatten(), outfile=f"{outdir}/icp_infl_{task}.png")
    #plot_corr(icp_scores.flatten(), ft_scores.flatten(), outfile=f"{outdir}/icp_ft_{task}.png")
    #plot_corr(infl_scores.flatten(), ft_scores.flatten(),  outfile=f"{outdir}/ft_infl_{task}.png")

    change_icp_scores = icp_scores - baseline_icp_scores
    change_ft_scores = ft_scores - baseline_ft_scores

    #plot_corr(np.mean(infl_scores, axis=0), np.mean(change_icp_scores, axis=0), outfile=f"{outdir}/icp_{task}.png")
    #plot_corr(np.mean(infl_scores, axis=0), np.mean(change_ft_scores, axis=0), outfile=f"{outdir}/ft_{task}.png")
    
    scatter_plot(infl_scores.flatten(), change_icp_scores.flatten(), 
                 x_label='Influence \n($\\nabla \ell(z)^T \\nabla \ell(x)$)',
                 y_label='$\Delta ICP$',
                 outfile=f"{outdir}/icp_{task}.png")

    scatter_plot(infl_scores.flatten(), change_ft_scores.flatten(), 
                 x_label='Influence \n($\\nabla \ell(z)^T \\nabla \ell(x)$)',
                 y_label='$\Delta ICP$',
                 outfile=f"{outdir}/ft_{task}.png")


if __name__ == "__main__":
    # same
    # different
    # commonsense_qa_answer_given_question_without_options
    # cnn_dailymail_3_0_0_news_summary
    # sciq_Direct_Question_Closed_Book_
    # social_i_qa_Generate_answer
    # cosmos_qa_context_answer_to_question
    # yelp_review_full_based_on_that

    task_types = ['alpaca_100',
                  'yelp_review_full_based_on_that',
                  'cnn_dailymail_3_0_0_news_summary',
                  'commonsense_qa_answer_given_question_without_options',
                  'cosmos_qa_context_answer_to_question',
                  'sciq_Direct_Question_Closed_Book_',
                  'social_i_qa_Generate_answer']
    
    #task_types = ['alpaca_100']

    outdir = '/home/cljiao/InContextDataValuation/outputs/change_plots_kmeans'
    for task in task_types:
        change_in_score(task=f"kmeans_{task}", outdir=outdir)

    #task_corr(task='yelp_review_full_based_on_that')
    #similar_different_task_corr()

"""
def diff_grad(task):
    icp_file = f"{data_dir}/icp_{task}_all.json"
    infl_file = f"{data_dir}/infl_{task}_all.json"

    icp_scores = read_file(icp_file)[0]
    infl_scores = read_file(infl_file)[0]

    k = 10000
    infl_scores_idxs = np.argsort(infl_scores)
    bottom_idxs = infl_scores_idxs[:k]
    top_idxs = infl_scores_idxs[-k:]

    top_infl = [infl_scores[i] for i in top_idxs]
    bottom_infl = [infl_scores[i] for i in bottom_idxs]

    top_icp = [icp_scores[i] for i in top_idxs]
    bottom_icp = [icp_scores[i] for i in bottom_idxs]

    plot_corr(top_icp,
              top_infl,
              '/home/cljiao/InContextDataValuation/outputs/plots/test_top.png',
              x_label='',
              y_label='',
              jitter=False)
    
    plot_corr(bottom_icp,
              bottom_infl,
              '/home/cljiao/InContextDataValuation/outputs/plots/test_bottom.png',
              x_label='',
              y_label='',
              jitter=False)

    diff = np.subtract(icp_scores, infl_scores)
    plt.scatter(infl_scores, diff, s=1, alpha=1)

    plt.xlabel('grad')
    plt.ylabel('diff')

    #plt.tick_params(labelleft=False, length=0)
    #plt.tick_params(labelbottom=False, length=0)
    #plt.yscale('log')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()

def similar_different_task_corr():
    ft_same_task_file = f"{data_dir}/ft_same_all.json"
    ft_different_task_file = f"{data_dir}/ft_different_all.json"
    icp_same_task_file = f"{data_dir}/icp_same_all.json"
    icp_different_task_file = f"{data_dir}/icp_different_all.json"
    infl_same_task_file = f"{data_dir}/infl_same_all.json"
    infl_different_task_file = f"{data_dir}/infl_different_all.json"

    icp_same_task = read_file(icp_same_task_file)
    ft_same_task = read_file(ft_same_task_file)
    infl_same_task = read_file(infl_same_task_file)

    icp_different_task = read_file(icp_different_task_file)
    ft_different_task = read_file(ft_different_task_file)
    infl_different_task = read_file(infl_different_task_file)

    print(get_avg_spearman(icp_same_task, infl_same_task))
    print(get_avg_spearman(icp_different_task, infl_different_task))

def plot_similar_different_tasks():
    ft_same_task_file = f"{data_dir}/ft_same_all.json"
    ft_different_task_file = f"{data_dir}/ft_different_all.json"
    icp_same_task_file = f"{data_dir}/icp_same_all.json"
    icp_different_task_file = f"{data_dir}/icp_different_all.json"
    infl_same_task_file = f"{data_dir}/infl_same_all.json"
    infl_different_task_file = f"{data_dir}/infl_different_all.json"

    #idx = 6
    icp_same_task = read_file(icp_same_task_file)
    ft_same_task = read_file(ft_same_task_file)
    infl_same_task = read_file(infl_same_task_file)

    icp_different_task = read_file(icp_different_task_file)
    ft_different_task = read_file(ft_different_task_file)
    infl_different_task = read_file(infl_different_task_file)

    get_avg_spearman(icp_same_task, infl_same_task)
    get_avg_spearman(icp_same_task, infl_different_task)

    plot_corr(icp_same_task, ft_same_task, f"{plots_dir}/icp_ft_same_task.png", x_label='ICP', y_label='FT')
    plot_corr(ft_same_task, infl_same_task, f"{plots_dir}/ft_infl_same_task.png", x_label='FT', y_label='Infl')
    plot_corr(icp_same_task, infl_same_task, f"{plots_dir}/icp_infl_same_task.png", x_label='ICP', y_label='Infl')

    plot_corr(icp_different_task, ft_different_task, f"{plots_dir}/icp_ft_different.png", x_label='ICP', y_label='FT')
    plot_corr(ft_different_task, infl_different_task, f"{plots_dir}/ft_infl_different.png", x_label='FT', y_label='Infl')
    plot_corr(icp_different_task, infl_different_task, f"{plots_dir}/icp_infl_different.png", x_label='ICP', y_label='Infl')
"""
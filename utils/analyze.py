"""
Various functions for analysis. Includes:
    - plot icp or influence score distributions
    - spearman correlation between icp and influence score
    - sort scores into ranking categories
    - overlaps between icp or influence scores
"""

import argparse
import json
import re
import os
from os.path import join
from collections import defaultdict

import numpy as np
import torch
from scipy import stats


data_dir = join(dirname(dirname(abspath(__file__))), "data/scores")

parser = argparse.ArgumentParser()
parser.add_argument("--icp_score_path",  default=join(data_dir, "data/scores/icp_scores.json"), type=str,
                                         help="path to icp scores")
parser.add_argument("--infl_score_path", default=join(data_dir, "data/scores/infl_ip.json"), type=str,
                                         help="path to influence scores")
args = parser.parse_args()


def read_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data


def read_infl_scores(path):
    scores = torch.load(path, map_location=torch.device('cpu')).numpy()
    scores = np.mean(scores, axis=0)
    return scores
    

def read_icp_scores(path):
    scores = read_json_file(path)
    scores = np.array([scores[str(i)][0] for i in range(len(scores))])
    return scores


def plot_score_distribution(scores, outfile, title='', xlabel='', ylabel='') -> None:
    """
    Sort scores and plot them

    Args:
        scores (List[float]): array of scores
        title (str, Optional): plot title
        xlabel (str, Optional): x-axis label
        ylabel (str, Optional): y-axis label
        outfile (str): path to save plot

    Returns:
        None
    """

    scores = np.sort(scores)
    x = list(range(len(scores)))
    plt.scatter(x, scores, s=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()


def parse_inequality(op):
    """
    Parse an inequality string

    Args:
         op (str): inequality string

    Returns:
        [op, n1, n2]: list of op (str), n1 (float), n2 (float)
    """

    # return [op, n1, n2]
    m = re.match(r'<=(0.[0-9]+)', op)
    if m:
        return ['<=', float(m.group(1)), None]

    m = re.match(r'>(0.[0-9]+)', op)
    if m:
        return ['>', float(m.group(1)), None]

    m = re.match(r'(0.[0-9]+)-([0-9]+.[0-9]+)', op)
    if m:
        return ['range', float(m.group(1)), float(m.group(2))]

    return None


def categorize_icp_scores(icp_scores, categories = ['<=0.5', '>0.5', '>0.6','>0.7', '>0.8', '>0.85', '>0.9', '>0.95']):
    """
    Categorize icp scores into score bins.

    Args:
        icp_scores (List[float]): icp scores
        categories (List[str]): score bins
    
    Returns:
        icp_bin_to_count (Dict[str, int]): number of prompts in each score bin
        icp_bin_to_prompt_id (Dict[str, List[int]]): mapping of prompt ids to icp score bins
    """

    icp_bin_to_prompt_id = defaultdict(list)
    parsed_categories = [parse_inequality(category) for category in categories]

    for prompt_id in range(len(icp_scores)):
        score = icp_scores[prompt_id]

        for category in categories:
            [op, n1, n2] = parse_category(category)

            if op == '<=' and score <= n1:
                icp_bin_to_prompt_id[category].append(prompt_id)
                
            if op == '>' and score > n1:
                icp_bin_to_prompt_id[category].append(prompt_id)

            if op == 'range' and n1 <= score <n2: 
                icp_bin_to_prompt_id[category].append(prompt_id)

    icp_bin_to_count = {category : len(icp_bin_to_prompt_id[category]) for category in icp_bin_to_prompt_id}
    return icp_bin_to_count, icp_bin_to_prompt_id


def find_overlaps(icp_bin_to_prompt_id, infl_scores):
    """
    Get overlapping points between influence and icp at different ranking bins.

    Args:
        icp_bin_to_prompt_id (Dict[str, str]): mapping of prompt ids to icp score bins
        infl_scores (List[float]): influence scores

    Returns: 
        None
    """

    for category, prompt_ids in icp_bin_to_prompt_id.items():
        if '<=' in category:
            infl_scores_idxs = np.argsort(infl_scores)
        else:
            infl_scores_idxs = np.argsort(infl_scores)[::-1]
        
        n = len(prompt_ids)
        infl_scores_idxs = set(infl_scores_idxs[:n])
        overlaps = infl_scores_idxs.intersection(prompt_ids)
        print(f"Overlaps for {category} score bin: {len(overlaps)}/{n}")


def icp_influence_spearman(icp_scores, influence_scores):
    """
    Spearman correlation between icp and influence scores.

    Args:
        icp_scores (List[float]): icp scores
        influence_scores (List[float]) influence_scores
    
    Returns
        None
    """

    res = stats.spearmanr(icp_scores, influence_scores)
    correlation = round(res.correlation, 3)
    pvalue = round(res.pvalue, 3)


if __name__ == "__main__":
    icp_scores = read_icp_scores(args.icp_score_path)
    infl_scores = read_infl_scores(args.infl_score_path)

    # plot icp score distribution
    plot_score_distribution(icp_scores,
                title='ICP Score Distribution', 
                xlabel='', 
                ylabel='Scores', 
                outfile='') # add plot outfile here

    # plot influence score distribution
    plot_score_distribution(infl_scores,
                title='Influence (IP) Distribution', 
                xlabel='', 
                ylabel='Scores', 
                outfile='') # add plot outfile here

    # spearman correlation of icp and influence scores
    icp_influence_spearman(icp_scores, infl_scores)

    # sort icp scores in rank bins (e.g., >0.9, >0.8 etc)
    categories_to_count, icp_bin_to_prompt_id = categorize_icp_scores(icp_scores)

    # number of overlaps between icp and influence in rank bins 
    find_overlaps(icp_bin_to_prompt_id, infl_scores)

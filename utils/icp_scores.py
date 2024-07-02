"""
Take model likelihood scores and calculate ICP scores
"""

import argparse
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--infile",  type=str, help="path to model likelihoods file")
parser.add_argument("--outfile", type=str, help="path to save file")
args = parser.parse_args()


def get_task_count(data):
    """
    Count the number of samples in the task.
 
    Args:
        data: model outputs.
 
    Returns:
        int: number of samples.
    """

    n_task_samples = set()
    for entry in tqdm(data):
        doc_id = entry['doc_id']
        n_task_samples.add(doc_id)
    return len(n_task_samples)


def load_data(path):
    """
    Data model outputs
 
    Args:
        path: model outputs.
 
    Returns:
        int: number of samples.
    """

    with open(path) as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data


def get_likelihoods(data):
    """
    Get output likelihoods from model outputs
 
    Args:
        data: model outputs
 
    Returns:
        likelihoods: model output likelihoods for each incontext sample
        baseline_likelihoods: baseline model output likelihoods (without incontext samples)
    """

    likelihoods = {}
    n_task_samples = get_task_count(data)
    baseline_likelihoods = np.zeros(n_task_samples)

    for entry in tqdm(data):
        incontext_doc_id = entry['incontext_doc_id']
        doc_id = entry['doc_id']
        ll = entry['filtered_resps'][0][0]

        # baseline case
        if incontext_doc_id == 0:
            baseline_likelihoods[doc_id] = ll
            continue

        incontext_doc_id = str(incontext_doc_id - 1) # minus one to exclude the baseline
        
        if incontext_doc_id not in likelihoods:
            likelihoods[incontext_doc_id] = np.zeros(n_task_samples)
        likelihoods[incontext_doc_id][doc_id] = ll

    return likelihoods, baseline_likelihoods


def get_icp_scores(likelihoods, baseline_likelihoods):
    """
    Calculate ICP scores.
 
    Args:
        likelihoods: model output likelihoods for each incontext sample
        baseline_likelihoods: baseline model output likelihoods (without incontext samples)
 
    Returns:
        icp scores
    """

    icp_scores = {}
    for incontext_doc_id in likelihoods:
        task_likelihoods = likelihoods[incontext_doc_id]
        icp_score = np.mean([1 if task_likelihoods[i] > baseline_likelihoods[i] else 0 for i in range(len(task_likelihoods))])
        icp_scores[incontext_doc_id] = [icp_score]
    return icp_scores
    

def save_icp_scores(icp_scores, path):
    with open(path, 'w+') as f:
        f.write(json.dumps(icp_scores))
    

if __name__ == "__main__":
    data = load_data(args.infile)
    likelihoods, baseline_likelihoods = get_likelihoods(data)
    icp_scores = get_icp_scores(likelihoods, baseline_likelihoods)
    save_icp_scores(icp_scores, args.outfile)
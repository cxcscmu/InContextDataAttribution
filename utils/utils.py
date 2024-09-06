import json
import torch
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join


def read_ft_scores(ft_path, n_train=None, rank=0):
    def match_format(text):
        return re.match(r'[0-9]+.pt', text)

    def get_number(text):
        m = re.match(r'([0-9]+).pt', text)
        return m.group(1)
    
    baseline_score = torch.load(f"{ft_path}/{rank}/baseline_scores.pt")
    
    ft_scores = []
    for i in range(n_train):
        score = torch.load(f"{ft_path}/{rank}/{i}.pt")[0]
        ft_scores.append(score)
    
    #return baseline_score + ft_scores
    return ft_scores


def read_icp_scores(ll_scores_path, n_train=None):
    with open(ll_scores_path) as f:
        data = [json.loads(l.strip()) for l in f.readlines()]

    icp_scores = np.zeros(n_train)
    baseline_score = None

    for entry in data:
        incontext_doc_id = entry['incontext_doc_id']
        ll = entry['filtered_resps'][0][0]

        if incontext_doc_id == 0:
            baseline_score = ll
            continue

        incontext_doc_id = incontext_doc_id-1
        icp_scores[incontext_doc_id] = ll
    
    icp_scores = icp_scores.tolist()
    #icp_scores.insert(0, baseline_score)
    return icp_scores


def read_inft_scores(infl_scores_path):
    return torch.load(infl_scores_path).tolist()[0]


if __name__ == "__main__":
    n_test, n_train = 50, 100
    outdir = '/home/cljiao/InContextDataValuation/outputs/gathered_scores'
    task_types = ['same', 'different']

    print('Collecting all icp scores')
    ll_dir = '/home/cljiao/InContextDataValuation/outputs/ll_scores'
    for task_type in task_types:
        all_icp_scores = []
        for i in tqdm(range(n_test)):
            ll_scores_path = f"{ll_dir}/{i}_{task_type}_tasks.json"
            icp_scores = read_icp_scores(ll_scores_path, n_train=n_train)
            all_icp_scores.append(icp_scores)

        with open(f"{outdir}/icp_{task_type}_all.json", 'w+') as f:
            f.write(json.dumps(all_icp_scores, indent=3))

    print('Collecting all ft scores')
    ft_dir = '/home/cljiao/InContextDataValuation/outputs/ft_scores'
    for task_type in task_types:
        all_ft_scores = []
        for i in tqdm(range(n_test)):
            ft_path = f"{ft_dir}/{i}_{task_type}_tasks"
            ft_scores = read_ft_scores(ft_path, n_train=n_train, rank=0)
            all_ft_scores.append(ft_scores)

        with open(f"{outdir}/ft_{task_type}_all.json", 'w+') as f:
            f.write(json.dumps(all_ft_scores, indent=3))

    print('Collecting all infl scores')
    infl_dir = '/home/cljiao/InContextDataValuation/outputs/infl_scores'
    for task_type in task_types:
        all_infl_scores = []
        for i in tqdm(range(n_test)):
            infl_scores_path = f"{infl_dir}/{i}_{task_type}_tasks.pt"
            infl_scores = read_inft_scores(infl_scores_path)
            all_infl_scores.append(infl_scores)

        with open(f"{outdir}/infl_{task_type}_all.json", 'w+') as f:
            f.write(json.dumps(all_infl_scores, indent=3))

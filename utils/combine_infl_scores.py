from os.path import dirname, abspath, join

def combine_infl_scores():
    """
    Example of how to combine paritions of influence scores
    """
    
    data_dir = join(dirname(dirname(abspath(__file__))), 'data/scores')
    files = [
        'infl_ip-1-of-9.pt',
        'infl_ip-2-of-9.pt',
        'infl_ip-3-of-9.pt',
        'infl_ip-4-of-9.pt',
        'infl_ip-5-of-9.pt',
        'infl_ip-6-of-9.pt',
        'infl_ip-7-of-9.pt',
        'infl_ip-8-of-9.pt',
        'infl_ip-9-of-9.pt'
    ]

    scores = []
    for filename in files:
        scores = torch.load(os.path.join(data_dir, filename),  map_location=torch.device('cpu')).numpy()
        scores.append(scores)

    scores = np.concatenate(scores, axis=1)
    torch.save(torch.tensor(scores), os.path.join(prefix, 'infl_ip.pt'))


if __name__ == "__main__":
    combine_infl_scores()
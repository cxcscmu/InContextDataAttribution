#!/bin/bash
#SBATCH --job-name=icp-all
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/icp-all.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/icp-all.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=5:00:00
#SBATCH --mem=16GB
#SBATCH --partition=general
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cljiao@andrew.cmu.edu
#SBATCH --exclude=babel-3-19

: '
name=icp_all
task="sciq_Direct_Question_Closed_Book_"
bash icp_template.sh \
     pretrained=EleutherAI/pythia-1b-deduped \
     /home/cljiao/heuristic-data/data/${task}.tsv \
     /home/cljiao/InContextDataValuation/outputs/ll_scores \
     ${task}

task="social_i_qa_Generate_answer"
bash icp_template.sh \
     pretrained=EleutherAI/pythia-1b-deduped \
     /home/cljiao/heuristic-data/data/${task}.tsv \
     /home/cljiao/InContextDataValuation/outputs/ll_scores \
     ${task}

task="cosmos_qa_context_answer_to_question"
bash icp_template.sh \
     pretrained=EleutherAI/pythia-1b-deduped \
     /home/cljiao/heuristic-data/data/${task}.tsv \
     /home/cljiao/InContextDataValuation/outputs/ll_scores \
     ${task}

task="yelp_review_full_based_on_that"
bash icp_template.sh \
     pretrained=EleutherAI/pythia-1b-deduped \
     /home/cljiao/heuristic-data/data/${task}.tsv \
     /home/cljiao/InContextDataValuation/outputs/ll_scores \
     ${task}
'

task="qa_dataset"
bash icp_template_alt.sh \
     pretrained=EleutherAI/pythia-1b-deduped \
     /home/cljiao/heuristic-data/data/${task}.tsv \
     /home/cljiao/InContextDataValuation/outputs/ll_kmeans/${task}.json
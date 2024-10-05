: '
name=one_kmeans_qa
task="doc_qa_dataset"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-1b \
       /data/user_data/cljiao/cont-pretrain/pythia-1b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task}
'

name="one_kmeans_self_instruct"
task="self_instruct"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-1b \
       /data/user_data/cljiao/cont-pretrain/pythia-1b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task}

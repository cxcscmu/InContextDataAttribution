name="minipile-1-of-10"
task="minipile-train-pythia-1b-256-n25000-1-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

name="minipile-2-of-10"
task="minipile-train-pythia-1b-256-n25000-2-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

name="minipile-3-of-10"
task="minipile-train-pythia-1b-256-n25000-3-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

name="minipile-4-of-10"
task="minipile-train-pythia-1b-256-n25000-4-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt


name="minipile-5-of-10"
task="minipile-train-pythia-1b-256-n25000-5-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

name="minipile-6-of-10"
task="minipile-train-pythia-1b-256-n25000-6-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

name="minipile-7-of-10"
task="minipile-train-pythia-1b-256-n25000-7-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

name="minipile-8-of-10"
task="minipile-train-pythia-1b-256-n25000-8-of-10"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_minipile_2.8/${name}.pt

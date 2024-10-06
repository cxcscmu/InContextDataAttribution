: '
name="alpaca-1-of-18"
task="alpaca-1-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-2-of-18"
task="alpaca-2-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-3-of-18"
task="alpaca-3-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-4-of-18"
task="alpaca-4-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt


name="alpaca-5-of-18"
task="alpaca-5-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-6-of-18"
task="alpaca-6-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-7-of-18"
task="alpaca-7-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-8-of-18"
task="alpaca-8-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-9-of-18"
task="alpaca-9-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-10-of-18"
task="alpaca-10-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-11-of-18"
task="alpaca-11-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-12-of-18"
task="alpaca-12-of-18"
sbatch --job-name=$name \
       --gres=gpu:8000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-13-of-18"
task="alpaca-13-of-18"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-14-of-18"
task="alpaca-14-of-18"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-15-of-18"
task="alpaca-15-of-18"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-16-of-18"
task="alpaca-16-of-18"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-17-of-18"
task="alpaca-17-of-18"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt

name="alpaca-18-of-18"
task="alpaca-18-of-18"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       EleutherAI/pythia-2.8b-deduped \
       EleutherAI/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_2.8/${task}.pt
'

name="alpaca-1-of-18"
task="alpaca-1-of-18"
bash infl_ip_template.sh \
       meta-llama/Llama-3.2-3B \
       meta-llama/Llama-3.2-3B \
       /home/cljiao/heuristic-data/alpaca_data/${task}.tsv \
       /data/user_data/cljiao/paper_outputs/infl_scores/kmeans_alpaca_llama/${task}.pt

name="kmeans"
sbatch --job-name=$name \
       --gres=gpu:A6000:4 \
       --time=1:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       icp_template.sh \
       kmeans100 \
       EleutherAI/pythia-12b-deduped \
       "/home/cljiao/InContextDataValuation/data/prompts/kmeans100.txt" \
       "/data/user_data/cljiao/one_step_12b/ll_scores/kmeans.json"

name="minipile"
sbatch --job-name=$name \
       --gres=gpu:A6000:4 \
       --time=1:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       icp_template.sh \
       minipile \
       EleutherAI/pythia-12b-deduped \
       "/data/user_data/cljiao/data-calibration/minipile-valid-pythia-1b-256-n100.tsv" \
       "/data/user_data/cljiao/one_step_12b/ll_scores/minipile.json"

name="kmeans-minipile"
sbatch --job-name=$name \
       --gres=gpu:A6000:4 \
       --time=1:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       icp_template.sh \
       kmeans100 \
       EleutherAI/pythia-12b-deduped \
       "/data/user_data/cljiao/data-calibration/minipile-valid-pythia-1b-256-n100.tsv" \
       "/data/user_data/cljiao/one_step_12b/ll_scores/kmeans-minipile.json"

name="mixture-minipile"
sbatch --job-name=$name \
       --gres=gpu:A6000:4 \
       --time=1:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       icp_template.sh \
       mixture \
       EleutherAI/pythia-12b-deduped \
       "/data/user_data/cljiao/data-calibration/minipile-valid-pythia-1b-256-n100.tsv" \
       "/data/user_data/cljiao/one_step_12b/ll_scores/mixture-minipile.json"

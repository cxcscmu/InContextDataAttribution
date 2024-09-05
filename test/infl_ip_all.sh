name="example"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=2-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       infl_ip_template.sh \
       cjiao/task-mixture-no-pythia \
       /data/user_data/cljiao/data-calibration/minipile-train-pythia-1b-256-n52000-20-of-20.tsv \
       /data/user_data/cljiao/pretrain/infl_mixture_optimal/scores-mixture-20-of-20.pt
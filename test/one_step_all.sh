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
'

name="one_kmeans_alpaca-5778"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       5778 \
       5778

name="one_kmeans_alpaca-11556"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       11556 \
       5778

name="one_kmeans_alpaca-17334"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       17334 \
       5778

name="one_kmeans_alpaca-23112"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       23112 \
       5778

name="one_kmeans_alpaca-28890"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       28890 \
       5778

name="one_kmeans_alpaca-34668"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       34668 \
       5778

name="one_kmeans_alpaca-40446"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       40446 \
       5778

name="one_kmeans_alpaca-46224"
task="kmeans_alpaca_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/alpaca_data/alpaca.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       46224 \
       5778

name="one_kmeans_minipile-0"
task="kmeans_minipile_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/minipile-train-pythia-1b-256-n25000.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       0 \
       5000

name="one_kmeans_minipile-5000"
task="kmeans_minipile_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/minipile-train-pythia-1b-256-n25000.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       5000 \
       5000

name="one_kmeans_minipile-10000"
task="kmeans_minipile_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/minipile-train-pythia-1b-256-n25000.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       10000 \
       5000

name="one_kmeans_minipile-15000"
task="kmeans_minipile_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/minipile-train-pythia-1b-256-n25000.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       15000 \
       5000

name="one_kmeans_minipile-20000"
task="kmeans_minipile_2.8"
sbatch --job-name=$name \
       --gres=gpu:A6000:1 \
       --time=1-00:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       one_step_template.sh \
       pythia-2.8b \
       /data/user_data/cljiao/cont-pretrain/pythia-2.8b-deduped \
       /home/cljiao/heuristic-data/train_data/minipile-train-pythia-1b-256-n25000.json \
       /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
       /data/user_data/cljiao/paper_outputs/ft_scores/${task} \
       20000 \
       5000
step_size=10
for i in `seq 0 $step_size 99`; 
do 
start=$i
end=$((i + step_size))
name=${start}-${end}
sbatch --job-name=$name \
       --gres=gpu:2080Ti:1 \
       --time=4:00:00 \
       --output="/home/cljiao/InContextDataValuation/logs/$name.out" \
       --error="/home/cljiao/InContextDataValuation/logs/$name.err" \
       bertscore_template.sh \
       $start \
       $end \
       "/data/user_data/cljiao/data-calibration/minipile-train-pythia-1b-256-n25000.tsv" \
       "/home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.tsv" \
       "/data/user_data/cljiao/paper_outputs/bertscores/kmeans_minipile"
done

python src/main.py \
	--model 'mlp' \
	--max_steps 100 \
	--dataset 'lung' \
	--use_best_hyperparams \
	--experiment_name 'test'  # GIVE A UNIQUE EXPERIMENT NAME EACH TIME, ELSE THE EXPERIMENTS WILL OVERLAP
	# --run_repeats_and_cv \  # if you want to runs 25 runs (5-fold cross-validation with 5 repeats) 
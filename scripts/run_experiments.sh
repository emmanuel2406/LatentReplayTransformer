export PYTHONPATH="$PYTHONPATH:$(pwd)/core:$(pwd)/small_100:$(pwd)/model:$(pwd)/dataset"
nohup python core/main.py --latent_layer_num 6 > nohup6.out

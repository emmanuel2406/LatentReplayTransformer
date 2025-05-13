export PYTHONPATH="$PYTHONPATH:$(pwd)/core:$(pwd)/small_100:$(pwd)/model:$(pwd)/dataset"
python core/inference.py --latent_layer_num 5 --rm_size 0 --temp 0.5
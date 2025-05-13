import argparse
import torch
from torch.nn import CrossEntropyLoss

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
import avalanche.models.utils

import sys
import os
from small_100.modeling_m2m_100 import M2M100Model, save_m2m100_model, load_m2m100_model
import model.utils

from core.latent_replay_transformer import LatentReplayTransformer as LRT
from core.load_flores200 import get_flores200_benchmark
from core.plotting import *

def get_strategy_config(latent_layer_num: int):
    device = torch.device(
        f"mps" if torch.backends.mps.is_available() else "cpu"
    )

    interactive_logger = InteractiveLogger()
    bleu_metric = model.utils.BLEUMetric()
    eval_plugin = EvaluationPlugin(
        bleu_metric,
        loggers=[interactive_logger]
    )

    return {
        "device": device,
        "criterion": CrossEntropyLoss(),
        "evaluator": eval_plugin,
        "lr": 5e-5 ,
        "train_mb_size": 1,
        "eval_mb_size": 4,
        "latent_layer_num": latent_layer_num,
        "max_seq_len": 128,
        "freeze_below_layer": f"encoder.layers.{latent_layer_num}.self_attn.k_proj",
        "warmup_steps": 4000,
        "train_epochs": 3,
        "pretrained": True,
    }

def train_with_eval(strategy, scenario):
    print('Starting experiment...')
    results = []

    seen_experiences = []
    for experience in scenario.train_stream:
        print(f"Start of experience {experience.current_experience}")
        seen_experiences.append(experience.current_experience)

        # Train on the current experience
        strategy.train(experience, num_workers=0)
        print('Training completed')

        # Evaluate on the corresponding test set
        print(f'Computing accuracy on test sets: {seen_experiences}')
        test_experiences = [scenario.test_stream[exp] for exp in seen_experiences]
        results.append(strategy.eval(test_experiences))
    save_m2m100_model(strategy.model, f"strategy_{strategy.latent_layer_num}_{strategy.rm_sz}")

    return results



def state_reset() -> None:
    """
    Reset model state.
    """
    print("Resetting model state...")
    with open("loss.txt", "w") as f:
        f.write("")
    with open("nohup.out", "w") as f:
        f.write("")

if __name__ == "__main__":
    state_reset()
    # command line arg for latent layer num
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_layer_num", type=int, default=5)
    parser.add_argument("--demo_subset", type=int, default=-1)
    args = parser.parse_args()
    strategy_config = get_strategy_config(args.latent_layer_num)

    # command line arg for number of examples
    # src_languages = ["fra_Latn", "ita_Latn", "spa_Latn", "afr_Latn", "deu_Latn"]  # Example source languages
    src_languages = ["afr_Latn", "xho_Latn", "zul_Latn", "tsn_Latn", "swh_Latn"] # Low-resouced languages
    # src_languages = ["fra_Latn", "ita_Latn", "spa_Latn", "afr_Latn"]
    # src_languages = ["fra_Latn", "ita_Latn"]
    tgt_language = "eng_Latn"

    demo_subset = None if args.demo_subset == -1 else args.demo_subset
    scenario = get_flores200_benchmark(src_languages, tgt_language, max_seq_len=128, demo_subset=demo_subset)

    # ablate for different rm_sizes:
    # rm_array = [50, 100, 250, 500]
    # rm_labels = []
    # for i in rm_array:
    #     for lang in src_languages:
    #         rm_labels.append(f"rm_array={i},{lang}")
    # for rm_sz in rm_array:
    #     replay_strategy = LRT(**strategy_config, rm_sz=rm_sz)
    #     replay_results = train_with_eval(replay_strategy, scenario)
    #     print("================EVALUATION RESULTS================")
    #     print(replay_results)

    # plot_loss_curve("loss.txt", strategy=f"latent replay({args.latent_layer_num})", labels=rm_labels)



    replay_strategy = LRT(**strategy_config, rm_sz=200)
    replay_results = train_with_eval(replay_strategy, scenario)
    with open("evaluation_results.txt", "a") as f:
        f.write(f"{args.latent_layer_num}:  {replay_results}\n")
    plot_loss_curve("loss.txt", strategy=f"latent replay({args.latent_layer_num})")

    # with open("loss.txt", "w") as f:
    #     f.write("")

    # vanilla_strategy = LRT(**strategy_config, rm_sz = 0)
    # vanilla_results = train_with_eval(vanilla_strategy, scenario)
    # with open("evaluation_results.txt", "a") as f:
    #     f.write(f"{args.latent_layer_num}:  {vanilla_results}\n")
    # plot_loss_curve("loss.txt", strategy="vanilla")
    # plot_eval_results(vanilla_results, replay_results)




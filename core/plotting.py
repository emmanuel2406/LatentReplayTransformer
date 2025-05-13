import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def plot_loss_curve(file_path, strategy:str, labels:list=None):
    """
    Reads a loss log file with format: iteration,loss and plots the loss curve.
    Automatically detects multiple training streams based on iteration resets.
    Parameters:
    - file_path (str): Path to the loss.txt file
    - strategy (str): Name of the strategy
    Returns:
    - None (displays and saves the plot)
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        id = random.randint(100, 1000)

    # Parse all loss streams
    loss_streams = []
    current_stream = []
    last_iter = -1

    for line in lines:
        if not line.strip():
            continue
        try:
            iter_str, loss_str = line.strip().split(",")
            iteration = int(iter_str)
            loss = float(loss_str)
            # New stream starts if iteration resets
            if iteration <= last_iter:
                loss_streams.append(current_stream)
                current_stream = []
            current_stream.append((iteration, loss))
            last_iter = iteration
        except ValueError:
            continue
    # Add the last stream if it has data
    if current_stream:
        loss_streams.append(current_stream)
    # If no streams were detected, exit
    if not loss_streams:
        print("PLOTTING: No valid loss data found in the file.")
        return
    print(f"PLOTTING: Detected {len(loss_streams)} task streams")

    if not labels:
        # Create default labels if not provided
        labels = [f"Stream {i+1}" for i in range(len(loss_streams))]

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Convert to DataFrames and plot each stream
    for i, stream_data in enumerate(loss_streams):
        stream_df = pd.DataFrame(stream_data, columns=["Iteration", "Loss"])
        plt.plot(
            stream_df["Iteration"], 
            stream_df["Loss"], 
            marker='o', 
            linewidth=2, 
            label=labels[i], 
            alpha=0.2
        )

    plt.title(f"Loss Curve for {strategy} of {len(loss_streams)} Tasks", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save and display the plot
    plt.savefig(f"images/loss_curve_{id}.png")



def plot_eval_results(vanilla_results, replay_results):
    """
    Plot BLEU score evaluation results for different languages over time.

    Parameters:
    -----------
    vanilla_results : list of dict
        A list of dictionaries where each dictionary maps language names to BLEU scores
        at a specific task time point.

    replay_results : list of dict
        A list of dictionaries with the same structure as vanilla_results,
        representing results from a different method.

    Returns:
    --------
    matplotlib.figure.Figure
        The plotted figure
    """
    id = random.randint(100, 1000)

    # Extract all languages from both result sets
    all_languages = set()
    for result_dict in vanilla_results + replay_results:
        for lang in result_dict.keys():
            all_languages.add(lang)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Define task time points (x-axis)
    vanilla_time_points = list(range(len(vanilla_results)))
    replay_time_points = list(range(len(replay_results)))

    # Define line styles and colors
    line_styles = {'vanilla': '-', 'replay': '--'}
    colors = plt.cm.tab10.colors  # Use colormap for different languages

    # Plot data for each language
    for i, language in enumerate(sorted(all_languages)):
        # Extract scores for vanilla results
        vanilla_scores = [result.get(language, np.nan) for result in vanilla_results]

        # Extract scores for replay results
        replay_scores = [result.get(language, np.nan) for result in replay_results]

        # Plot vanilla results
        plt.plot(
            vanilla_time_points, 
            vanilla_scores, 
            label=f"{language} (vanilla)", 
            color=colors[i % len(colors)], 
            linestyle=line_styles['vanilla'],
            marker='o',
            linewidth=2
        )
        # Plot replay results
        plt.plot(
            replay_time_points, 
            replay_scores, 
            label=f"{language} (replay)", 
            color=colors[i % len(colors)], 
            linestyle=line_styles['replay'],
            marker='s',
            linewidth=2
        )
    # Add labels and title
    plt.xlabel('Task Time', fontsize=14)
    plt.ylabel('BLEU Score', fontsize=14)
    plt.title('BLEU Score Comparison: Vanilla vs. Replay Methods', fontsize=16)
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    # Create legend
    plt.legend(loc='best', fontsize=12)
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"images/bleu_comparison_{id}.png")
    return plt.gcf()

def plot_latent_layer_num(data, lang_idx:int):
    """
    Plots a bar graph comparing BLEU scores across different tasks for various latent layers.
    
    Parameters:
        data (dict): Keys are latent layer labels, values are lists of BLEU scores for each task.
    """
    # Extract task names (assumed to be indices here)
    num_tasks = len(next(iter(data.values())))
    task_indices = np.arange(num_tasks)
    task_languages = ["Afrikaans(1)", "Xhosa(2)", "Zulu(3)", "Tswana(4)", "Swahili(5)"]
    bar_width = 0.2
    grey_shades = ['#2F4F4F', '#708090', '#A9A9A9', '#D3D3D3']

    # Plot setup
    plt.figure(figsize=(10, 6))

    # Offset for each latent layer's bars
    for i, (layer_name, scores) in enumerate(data.items()):
        offset = (i - len(data)/2) * bar_width + bar_width/2
        color = grey_shades[i % len(grey_shades)]
        plt.bar(task_indices + offset, scores, width=bar_width, label=layer_name, color=color)

    # Labels and legend
    plt.xlabel("Tasks in chronological order", fontsize=14)
    plt.ylabel("BLEU Score", fontsize=14)
    plt.xticks(task_indices, task_languages[lang_idx:], fontsize=12)
    plt.title(f"BLEU Score Comparison for {task_languages[lang_idx]} across different tasks", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(f"images/latent_layer_{lang_idx}.png")

if __name__ == "__main__":
    afrikaans_data = {
        "latent layer 1": [
            0.011380852559501248,
            0.059289797694361476,
            0.158971788468371,
            0.19471584011035517,
            0.16807866499102672
        ],
        "latent layer 3": [
            0.12918262320863302,
            0.19806917924832151,
            0.18917173327265094,
            0.2128799503785885,
            0.18828500204697052
        ],
        "latent layer 7": [
            0.14846670143570903,
            0.2092598824474746,
            0.2077591864949021,
            0.19785085683486667,
            0.18841003814902335
        ],
        "latent layer 11": [
            0.07784092974347062,
            0.14460969901771428,
            0.15690091083778923,
            0.15643356364353417,
            0.164084797194956
        ]
    }
    xhosa_data = {
        "latent layer 1": [
            0.059263411198605,
            0.15897570109547193,
            0.19471584011035517,
            0.16807934537368396,
        ],
        "latent layer 3": [
            0.19410633298876426,
            0.18256628710088815,
            0.2127170465785026,
            0.19016015170620812
        ],
        "latent layer 7": [
            0.20230376672437797,
            0.20505286789723087,
            0.1996931403948941,
            0.187814021968577,
        ],
        "latent layer 11": [
            0.14251688971795415,
            0.15136105467337038,
            0.14060860662629618,
            0.1614282253213533
        ]
    }
    zulu_data = {
        "latent layer 1": [
            0.1591804657999243,
            0.19471584011035517,
            0.16807866499102672
        ],
        "latent layer 3": [
            0.18307270999813816,
            0.21297944732482907,
            0.19032201506929716,
        ],
        "latent layer 7": [
           0.20528201788008849,
           0.19938651275351135,
           0.18777510215939638
        ],
        "latent layer 11": [
            0.1523092046527234,
            0.14021812428014174,
            0.16194264391585628
        ]
    }
    plot_latent_layer_num(afrikaans_data, 0)
    plot_latent_layer_num(xhosa_data, 1)
    plot_latent_layer_num(zulu_data, 2)
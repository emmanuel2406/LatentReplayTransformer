from torch.utils.data import Dataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.scenarios import NCScenario
from avalanche.benchmarks.utils import make_avalanche_dataset

from torch.nn.utils.rnn import pad_sequence
import torch

import sys
import os

from small_100.tokenization_small100 import SMALL100Tokenizer
from dataset.flores_200 import Flores200, Flores200Config

from torch.utils.data import DataLoader

class FloresDataset(Dataset):
    def __init__(self, data_dir, src_lang, tgt_lang, suffix, demo_subset=None):
        self.data = []
        self.targets = []
        src_file_path = os.path.join(data_dir, suffix,  f"{src_lang}.{suffix}")
        tgt_file_path = os.path.join(data_dir, suffix,  f"{tgt_lang}.{suffix}")

        if not os.path.exists(src_file_path):
            raise FileNotFoundError(f"Source file {src_file_path} does not exist.")
        if not os.path.exists(tgt_file_path):
            raise FileNotFoundError(f"Target file {tgt_file_path} does not exist.")

        with open(src_file_path, "r", encoding="utf-8") as src_file, \
             open(tgt_file_path, "r", encoding="utf-8") as tgt_file:
            src_sentences = src_file.readlines()
            tgt_sentences = tgt_file.readlines()

            if len(src_sentences) != len(tgt_sentences):
                raise ValueError("Source and target files must have the same number of lines.")

            if demo_subset is not None:
                src_sentences = src_sentences[:demo_subset]
                tgt_sentences = tgt_sentences[:demo_subset]

            self.data.extend(src_sentences)
            self.targets.extend(tgt_sentences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].strip(), self.targets[idx].strip()


"""
Convert the flores code into the small-100 tokenizer code
"""
def convert_code(flores_code: str) -> str:
    # for now consider the simple rule of taking the first two chars
    irregular_codes = {"spa_Latn": "es", "tsn_Latn": "tn"}
    if flores_code in irregular_codes.keys():
        return irregular_codes[flores_code]
    return flores_code[:2]


# Create a picklable class instead of a closure
class FloresCollateFunction:
    def __init__(self, src_lang, tgt_lang, max_seq_len=128):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        src_texts, tgt_texts = zip(*batch)
        print(f"Collate executed for {self.src_lang} -> {self.tgt_lang}")
        tokenizer = SMALL100Tokenizer()
        tokenizer.src_lang = convert_code(self.src_lang)
        tokenizer.tgt_lang = convert_code(self.tgt_lang)

        src_tokens = [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in src_texts]
        tgt_tokens = [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in tgt_texts]

        def pad_or_truncate(tokens):
            """Pads or truncates a sequence to `max_seq_len`."""
            if len(tokens) < self.max_seq_len:
                return torch.cat([tokens, torch.full((self.max_seq_len - len(tokens),), tokenizer.pad_token_id)])
            else:
                print(f"TRUNCATING WARNING:{len(tokens)}")
                return tokens[:self.max_seq_len]  # Truncate if too long

        src_padded = torch.stack([pad_or_truncate(tokens) for tokens in src_tokens])
        tgt_padded = torch.stack([pad_or_truncate(tokens) for tokens in tgt_tokens])

        # Create attention masks (1 for real tokens, 0 for padding)
        src_attention_mask = (src_padded != tokenizer.pad_token_id).long()
        tgt_attention_mask = (tgt_padded != tokenizer.pad_token_id).long()


         # [:, 0, :]-> encoder input, [:, 1, :]-> decoder input
        return {
            "input_ids": torch.stack([src_padded, tgt_padded], dim=1),
            "attention_mask": torch.stack([src_attention_mask, tgt_attention_mask], dim=1)
        }


def create_collate_fn(src_lang, tgt_lang, max_seq_len):
    return FloresCollateFunction(src_lang, tgt_lang, max_seq_len)


def get_flores200_benchmark(src_languages: list[str], tgt_language: str, max_seq_len=128, demo_subset=None)->NCScenario:
    # Define the directory containing your .dev files and the languages of interest
    data_directory = os.path.join("dataset", "flores200_dataset")
    train_suffix = "dev"
    test_suffix = "devtest"

    train_datasets = []
    test_datasets = []

    for task_id, src_lang in enumerate(src_languages):
        train_flores = FloresDataset(data_directory, src_lang, tgt_language, train_suffix, demo_subset=demo_subset)
        test_flores = FloresDataset(data_directory, src_lang, tgt_language, test_suffix, demo_subset=demo_subset)

        lang_specific_collate_fn = create_collate_fn(src_lang, tgt_language, max_seq_len)
        train_flores.collate_fn = lang_specific_collate_fn
        test_flores.collate_fn = lang_specific_collate_fn

        wrapped_train = AvalancheDataset(train_flores)
        wrapped_train.src_lang = src_lang
        wrapped_train.tgt_lang = tgt_language

        wrapped_test = AvalancheDataset(test_flores)
        wrapped_test.src_lang = src_lang
        wrapped_test.tgt_lang = tgt_language

        train_datasets.append(wrapped_train)
        test_datasets.append(wrapped_test)

    scenario = benchmark_from_datasets(train=train_datasets, test=test_datasets)

    scenario.src_languages = src_languages
    scenario.tgt_language = tgt_language

    return scenario


if __name__ == "__main__":
    # Set up languages for testing
    src_languages = ["eng_Latn", "ita_Latn", "afr_Latn"]  # Example source languages
    tgt_language = "fra_Latn"
    
    print("=== Creating FLORES200 Benchmark ===")
    flores_scenario = get_flores200_benchmark(src_languages, tgt_language)
    print(f"Number of training experiences: {len(flores_scenario.train_stream)}")
    print(f"Number of test experiences: {len(flores_scenario.test_stream)}")
    
    print("\n=== Testing Collate Function with DataLoader ===")
    for experience in flores_scenario.train_stream:
        print(f"Experience {experience.current_experience} - Dataset size: {len(experience.dataset)}")
        print(f"Source language: {src_languages[experience.current_experience]}")
        
        # Get a few raw samples from the dataset for comparison
        print("\nRaw samples before collate_fn:")
        for i in range(min(2, len(experience.dataset))):
            raw_sample = experience.dataset[i]
            if isinstance(raw_sample, tuple) and len(raw_sample) >= 2:
                print(f"  Source text: {raw_sample[0][:50]}...")
                print(f"  Target text: {raw_sample[1][:50]}...")
            else:
                print(f"  Sample content: {raw_sample}")
        
        # Create a DataLoader with custom collate function
        print("\nCreating DataLoader with tokenizer_collate_fn...")
        experience_collate_fn = create_collate_fn(src_languages[experience.current_experience], tgt_language)
        dataloader = DataLoader(
            experience.dataset,
            batch_size=2,  # Using batch_size=2 to see batching in action
            shuffle=True,
            collate_fn=experience_collate_fn
        )
        
        # Get a batch and verify structure
        print("\nFetching first batch from DataLoader...")
        try:
            batch = next(iter(dataloader))
            print(f"Batch type: {type(batch)}")
            print(f"Batch shape: {batch.shape if hasattr(batch, 'shape') else 'N/A'}")
            
            # Print details of first few samples in the batch
            print("\nProcessed samples after collate_fn:")
            for i in range(min(batch.size(0), 2)):
                print(f"Sample {i}:")
                print(f"  Shape: {batch[i].shape}")
                
                # Print source tokens (first dimension)
                src_tokens = batch[i, 0, :]
                print(f"  Source tokens: {src_tokens[:10]}...")
                
                # Print target tokens (second dimension)
                tgt_tokens = batch[i, 1, :]
                print(f"  Target tokens: {tgt_tokens[:10]}...")
                
                # Attempt to decode a few tokens for verification
                tokenizer = SMALL100Tokenizer()
                print(f"  First few source tokens decoded: {tokenizer.decode(src_tokens[:5])}")
                print(f"  First few target tokens decoded: {tokenizer.decode(tgt_tokens[:5])}")
            
        except Exception as e:
            print(f"Error when fetching batch: {e}")
            import traceback
            traceback.print_exc()
"""
    Test out translations of fine-tuned model checkpoints
"""
import argparse
import os
import torch

from small_100.modeling_m2m_100 import load_m2m100_model
from core.load_flores200 import convert_code
from small_100.tokenization_small100 import SMALL100Tokenizer



class TranslationRequest:
    def __init__(self, user_input: str, source_language: str, target_language: str, temp: float):
        self.user_input = user_input
        self.src_lang = source_language
        self.tgt_lang = target_language
        self.temp = temp
        self.max_seq_len = 128
        self.tokenizer = SMALL100Tokenizer()
        self.tokenizer.src_lang = convert_code(self.src_lang)
        self.tokenizer.tgt_lang = convert_code(self.tgt_lang)

    def preprocess_input(self):
        src_tokens = [self.tokenizer(self.user_input, return_tensors="pt")["input_ids"][0]]
        def pad_or_truncate(tokens):
            """Pads or truncates a sequence to max_seq_len."""
            if len(tokens) < self.max_seq_len:
                return torch.cat([tokens, torch.full((self.max_seq_len - len(tokens),), self.tokenizer.pad_token_id)])
            else:
                print(f"TRUNCATING WARNING:{len(tokens)}")
            return tokens[:self.max_seq_len] # Truncate if too long
        src_padded = torch.stack([pad_or_truncate(tokens) for tokens in src_tokens])
        # Create attention masks (1 for real tokens, 0 for padding)
        src_attention_mask = (src_padded != self.tokenizer.pad_token_id).long()
        # [:, 0, :]-> encoder input, [:, 1, :]-> decoder input
        return {
            "input_ids": src_padded,
            "attention_mask": src_attention_mask
        }


@torch.no_grad()
def translate_op(model, request: TranslationRequest):
    input_data = request.preprocess_input()
    
    # Get the tokenizer from the request
    tokenizer = request.tokenizer
    
    decoder_start_token_id = request.tokenizer.get_lang_id(convert_code(request.tgt_lang))
    decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

    # Encoder outputs (compute once)
    encoder_outputs = model.encoder(
        input_ids=input_data["input_ids"],
        attention_mask=input_data["attention_mask"],
        return_dict=True
    )

    # Generate tokens one by one
    for _ in range(request.max_seq_len):
        # Get model outputs
        print(decoder_input_ids.shape)
        outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=input_data["attention_mask"],
            return_dict=True
        )

        # Get logits from the final layer
        logits = model.lm_head(outputs.last_hidden_state)

        print("Logits shape", logits.shape)
        # Get the logits for the next token (last position)
        next_token_logits = logits[:, -1, :]
        next_token_logits[0, decoder_start_token_id] = -float("inf")
        # Apply temperature sampling
        if request.temp == 0 or request.temp is None:
            # Temperature of 0 means greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            # Scale logits by temperature
            scaled_logits = next_token_logits / max(float(request.temp), 1e-7)
            # Convert to probabilities
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
        print("next token", next_token)
        # Append the generated token to the sequence
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
        # Stop if we generate the EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode the generated sequence
    translation = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
    return translation



def create_translation_session(latent_layer_num: int, rm_size: int, temp: float=0.0):
    load_path = f"strategy_{latent_layer_num}_{rm_size}"
    model = load_m2m100_model(load_path=load_path, latent_layer_num=latent_layer_num)
    model.eval()

    user_input = " "
    source_language = "fra_Latn"# source_language = input("Source language:")
    print("Enter blank to end session")
    while user_input != "":
        user_input = "Je m'appelle Emmanuel, et j'ai vingt ans cette année"# user_input = input("Text to translate:")
        target_language = "eng_Latn" # by default assume target = eng_Latn
        request = TranslationRequest(user_input, source_language, target_language, temp)

        translation = translate_op(model, request)
        print("Translation:", translation)


if __name__ == "__main__":
    # command line arg for latent layer num
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_layer_num", type=int, default=5)
    parser.add_argument("--rm_size", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    create_translation_session(args.latent_layer_num, args.rm_size, args.temp)


from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tokenization_small100 import SMALL100Tokenizer

hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
chinese_text = "生活就像一盒巧克力。"
af_text = "Nee dis nie jou fout nie."
fr_text = "La vie est comme une boîte de chocolat."

SIZE = "small"

if SIZE == "small":
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    tokenizer = SMALL100Tokenizer()
elif SIZE == "medium":
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate Hindi to French
tokenizer.src_lang = "hi"
tokenizer.tgt_lang = "en"
encoded_hi = tokenizer(hi_text, return_tensors="pt")
# print(encoded_hi)
# print(tokenizer.decode(encoded_hi["input_ids"][0]))
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr"])
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => "La vie est comme une boîte de chocolat."


# translate Afrikaans to English
tokenizer.src_lang = "af"
tokenizer.tgt_lang = "en"
encoded_af = tokenizer(af_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_af, forced_bos_token_id=tokenizer.lang_code_to_id["en"])
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => "No, it is not your mistake."
"""Load a 4-bit quantised chat model and generate greedy answers (needs a GPU; imports are lazy
so that the rest of the package, and CI, work without torch)."""

from __future__ import annotations

import time

MAX_NEW_TOKENS = 400


def load_llm(name: str):
    """Return (tokenizer, model, loader_name). NF4 4-bit with fp16 compute: a T4 has no bf16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=quant, device_map="auto", dtype=torch.float16
        )
        return tokenizer, model, "AutoModelForCausalLM"
    except Exception as error:  # noqa: BLE001  some multimodal models are not causal LMs
        print(f"AutoModelForCausalLM failed for {name}: {error!r}; trying ImageTextToText")
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            name, quantization_config=quant, device_map="auto", dtype=torch.float16
        )
        return tokenizer, model, "AutoModelForImageTextToText"


def generate(tokenizer, model, messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Greedy decoding with thinking switched off. Returns text, token counts and seconds."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "new_tokens": len(new_tokens),
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "seconds": round(time.time() - start, 1),
    }


import time
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_MAP = {
    "LOW": "Qwen/Qwen2.5-0.5B-Instruct",
    "MEDIUM": "Qwen/Qwen2.5-1.5B-Instruct",
    "HIGH": "Qwen/Qwen2.5-3B-Instruct",
}

COST_MULTIPLIER = {
    "LOW": 1.0,
    "MEDIUM": 2.0,
    "HIGH": 4.0,
}

_model = None
_tokenizer = None
_loaded_tier = None


def _clear_model():
    global _model, _tokenizer, _loaded_tier

    if _model is not None:
        del _model

    if _tokenizer is not None:
        del _tokenizer

    _model = None
    _tokenizer = None
    _loaded_tier = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(tier):
    global _model, _tokenizer, _loaded_tier

    if tier not in MODEL_MAP:
        raise ValueError(f"Unknown tier: {tier}")

    if _loaded_tier == tier and _model is not None:
        return

    # Only keep one model in GPU memory.
    _clear_model()

    model_name = MODEL_MAP[tier]

    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda"
    )

    _model.eval()
    _loaded_tier = tier


def generate_response(query, tier, max_new_tokens=256):
    """
    Generate a real local LLM response.

    Returns:
        response
        router-independent LLM generation latency
        input token count
        output token count
        total token count
        model name
        normalized cost multiplier
    """

    load_model(tier)

    messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    input_tokens = inputs["input_ids"].shape[1]

    # Warm-up only when needed.
    if not hasattr(generate_response, "_warmed_tier"):
        generate_response._warmed_tier = {}

    if tier not in generate_response._warmed_tier:
        with torch.inference_mode():
            _model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False
            )

        torch.cuda.synchronize()
        generate_response._warmed_tier[tier] = True

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    torch.cuda.synchronize()

    end = time.perf_counter()

    latency = end - start

    generated_ids = output_ids[0][input_tokens:]

    response = _tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    output_tokens = len(generated_ids)

    return {
        "response": response,
        "model_name": MODEL_MAP[tier],
        "llm_latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "normalized_cost": COST_MULTIPLIER[tier],
    }

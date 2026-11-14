def estimate_input_tokens(prompt: str) -> int:
    """Approximate the prompt's token count by counting whitespace-separated words.

    Used only by the simulated inference engine, which has no tokenizer. The Hugging Face
    engine counts real tokens with the model's tokenizer and chat template.
    """
    return len(prompt.split())

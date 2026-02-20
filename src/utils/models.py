from transformers import WhisperForConditionalGeneration


def load_base_model(model_name, processor, bitfit=True, train_layernorm=True):
    """
    Load Whisper model and optionally apply BitFit (train only bias + LN).
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}",
        device_map="auto"
    )

    # Generation config
    model.generation_config.language = processor.tokenizer.language
    model.generation_config.task = processor.tokenizer.task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=processor.tokenizer.language,
        task=processor.tokenizer.task
    )

    if bitfit:
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze bias (+ optionally LayerNorm)
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            if train_layernorm and ("layer_norm" in name or "layernorm" in name):
                param.requires_grad = True

    return model
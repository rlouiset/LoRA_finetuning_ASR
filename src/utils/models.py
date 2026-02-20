from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, get_peft_model, PrefixTuningConfig, TaskType


def load_base_model(model_name, processor):
    """
    Load the base Whisper model and set generation config.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}",
        device_map="auto"
    )

    # Set generation config for transcription
    model.generation_config.language = processor.tokenizer.language
    model.generation_config.task = processor.tokenizer.task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=processor.tokenizer.language,
        task=processor.tokenizer.task
    )

    return model


def load_prefix_model(model_name, processor, prefix_path):
    """
    Load Whisper base model and apply Prefix adapters for inference.
    """
    model = load_base_model(model_name, processor)
    model = PeftModel.from_pretrained(model, prefix_path)
    return model


def prepare_prefix_for_training(config, processor):
    """
    Load base model and wrap it with Prefix Tuning adapters for fine-tuning.
    """

    # Load base model
    model = load_base_model(config.whisper_model, processor)

    # Prefix tuning configuration
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,      # Whisper is encoder-decoder
        num_virtual_tokens=20,
        encoder_hidden_size=model.config.d_model
    )

    # Wrap with PEFT prefix adapters
    model = get_peft_model(model, prefix_config)

    model.print_trainable_parameters()

    return model
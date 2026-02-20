from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import get_peft_model, PromptTuningConfig, TaskType, PeftModel


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


def load_prompt_model(model_name, processor, prompt_path):
    """
    Load Whisper base model and apply Prompt Tuning adapters for inference.
    """
    model = load_base_model(model_name, processor)
    model = PeftModel.from_pretrained(model, prompt_path)
    return model


def prepare_prompt_for_training(config, processor):
    """
    Load base model and wrap it with Prompt Tuning adapters for fine-tuning.
    """

    # Load base model
    model = load_base_model(config.whisper_model, processor)

    # Prompt tuning configuration
    prompt_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # Whisper is encoder-decoder
        num_virtual_tokens=50             # number of learnable tokens; can tune
    )

    # Wrap model with PEFT prompt adapters
    model = get_peft_model(model, prompt_config)

    model.print_trainable_parameters()

    return model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, get_peft_model, IA3Config


def load_base_model(model_name, processor):
    """
    Load the base Whisper model and set generation config.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}",
        # Use single-GPU / auto device placement
        device_map=None  # We'll handle device placement in Trainer
    )

    # Set generation config for transcription
    model.generation_config.language = processor.tokenizer.language
    model.generation_config.task = processor.tokenizer.task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=processor.tokenizer.language,
        task=processor.tokenizer.task
    )

    return model


def load_ia3_model(model_name, processor, ia3_path):
    """
    Load Whisper base model and apply IA3 adapters for inference.
    """
    model = load_base_model(model_name, processor)
    model = PeftModel.from_pretrained(model, ia3_path)
    return model


def prepare_ia3_for_training(config, processor):
    """
    Load base model and wrap it with IA3 adapters for fine-tuning.
    """

    # IA3 configuration
    ia3_config = IA3Config(
        target_modules=config.target_modules,  # e.g., ["q_proj", "v_proj"]
        task_type="SEQ_2_SEQ_LM",              # required for Hugging Face seq2seq
        feedforward_modules=["fc1", "fc2"],           # MLP layers in Whisper
    )

    # Load base model
    model = load_base_model(config.whisper_model, processor)

    # Wrap with IA3 adapters
    model = get_peft_model(model, ia3_config)

    return model

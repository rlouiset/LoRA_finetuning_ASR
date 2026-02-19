from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel,get_peft_model, LoraConfig


def load_base_model(model_name, processor):
    """Load the base Whisper model and set generation config."""
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}", 
        device_map="auto"
    )
    model.generation_config.language = processor.tokenizer.language
    model.generation_config.task = processor.tokenizer.task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=processor.tokenizer.language,
        task=processor.tokenizer.task
    )

    return model

def load_lora_model(model_name, processor, lora_path):
    """Load Whisper model and apply LoRA weights for inference."""
    model = load_base_model(model_name, processor)
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def prepare_lora_for_training(config, processor):
    """Load base model and wrap it with LoRA for fine-tuning."""
    
    # Lora Configuration
    lora_config = LoraConfig(
        r=config.lora_r,  # Rank 1
        lora_alpha=config.lora_alpha,  # Scaling factor 
        target_modules=config.target_modules,  # Attention modules to adapt
        lora_dropout=config.lora_dropout,  # dropout 
        bias="none",  # no bias adaptation
  
    )
    model = load_base_model(config.whisper_model, processor)
    model = get_peft_model(model, lora_config)
    return model

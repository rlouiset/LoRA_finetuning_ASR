import numpy as np
import string


def make_compute_metrics(processor, metric):
    def compute_metrics(eval_pred):
        """
        Function called by the trainer to compute the WER metric
        """
        predictions, labels = eval_pred

        print("Predictions type:", type(predictions))
        predictions = np.where(predictions != -100, predictions, processor.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        # Get rid of punctuation
        decoded_preds = [normalized_ponctuation(t) for t in decoded_preds]
        decoded_labels = [normalized_ponctuation(t) for t in decoded_labels]


        # WER computing
        wer_score = 100*metric.compute(predictions=decoded_preds,references=decoded_labels)


        return {"eval_wer": wer_score}
    
    return compute_metrics


#we normalized punctuation because orthophonist's transcription has no punctuation while Whisper has




def normalized_ponctuation(text: str) -> str:
    # we keep "'" and "-" signs for meaning
    ponctuation_sans_tiret = string.punctuation.replace("-", "").replace("'", "")
    ref_text_sans_ponct = "".join(c for c in text.lower() if c not in ponctuation_sans_tiret)

    return ref_text_sans_ponct
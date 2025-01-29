from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import mean_squared_error
import torch
import wandb

from ..base import BasePlugin
from .utils import text_to_df


class ValidationLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]
        print(f"ValidationLossCallback on_step_end: {state.global_step}")
        if state.global_step % args.eval_steps == 0:  # Evaluate every `eval_steps`
            eval_dataloader = trainer.get_eval_dataloader()
            model = trainer.model
            model.eval()
            try:
                # Compute simple validation metrics without external libraries
                total_exact_matches = 0
                total_samples = 0
                total_mse = 0
                total_samples_for_mse = 0
                print("Starting validation here")
                with torch.no_grad():

                    for batch in eval_dataloader:
                        batch = {k: v.to(args.device) for k, v in batch.items()}

                        # Generate sequences
                        generated_tokens = model.generate(
                            input_ids=batch["input_ids"],
                            max_length=256
                        )

                        # Convert generated token IDs to text
                        predicted_texts = trainer.tokenizer.batch_decode(
                            generated_tokens,
                            skip_special_tokens=True
                        )
                        # Convert reference token IDs (labels) to text  
                        reference_texts = trainer.tokenizer.batch_decode(
                            batch["labels"],
                            skip_special_tokens=True
                        )
                        print("------------------")
                        print(predicted_texts)
                        preds = text_to_df(predicted_texts)
                        print("~~actual~~")
                        print(reference_texts)
                        actual = text_to_df(reference_texts)
                        print("------------------")

                        # Get the minimum length between predicted and actual blood glucose values
                        min_length = min(len(preds.blood_glucose), len(actual.blood_glucose))
                        min_length = min(min_length, 12)
                        # Truncate both series to the minimum length
                        preds_bg = preds.blood_glucose.iloc[:min_length]
                        actual_bg = actual.blood_glucose.iloc[:min_length]
                        # Add to total MSE for later aggregation
                        total_mse += mean_squared_error(actual_bg, preds_bg, squared=True) 
                        total_samples_for_mse += len(actual_bg)
                                               
                        # Count exact matches
                        for pred, ref in zip(predicted_texts, reference_texts):
                            if pred.strip() == ref.strip():
                                total_exact_matches += 1
                            total_samples += 1

                wandb.log({"rmse": (total_mse / total_samples_for_mse) ** 0.5}, step=state.global_step)

                exact_match_accuracy = total_exact_matches / total_samples if total_samples > 0 else 0
                print(f"Validation exact match accuracy at step {state.global_step}: {exact_match_accuracy:.4f}")
                wandb.log({"exact_match_accuracy": exact_match_accuracy}, step=state.global_step)
            except Exception as e:
                print(f"Error in ValidationLossCallback: {e}")
            finally:
                print("Switching back to training mode")
                model.train()  # Switch back to training mode


class ValidationLossPlugin(BasePlugin):
    def on_train_begin(self, trainer):
        trainer.add_callback(ValidationLossCallback())
    
    def get_input_args(self):
        return "axolotl.integrations.grokfast.GrokfastArgs"

    def add_callbacks_pre_trainer(self, cfg, trainer):
        return []

    def add_callbacks_post_trainer(self, cfg, trainer):
        callback = ValidationLossCallback()
        return [callback]


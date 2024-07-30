from cog import BasePredictor, Input, Path
import torch

from transformers import AutoModelForCausalLM
from anticipation.sample import generate, generate_ar

# Assuming all other required imports and functions (like generate and generate_ar) are defined in the same file or imported appropriately

#MODELS = ["stanford-crfm/music-medium-800k", "stanford-crfm/music-small-800k", "stanford-crfm/music-large-800k"]
#MODELS = ["stanford-crfm/music-medium-800k", "stanford-crfm/music-small-800k"]
MODELS = ["stanford-crfm/music-small-800k", "stanford-crfm/music-medium-800k", 'stanford-crfm/music-large-800k']
#MODELS = ["stanford-crfm/music-large-800k", "stanford-crfm/music-medium-800k", 'stanford-crfm/music-small-800k']

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Model should be loaded here with the appropriate configuration
        self.models = {}

    def _load_model(self, m):
        if m not in self.models:
            self.models[m] = AutoModelForCausalLM.from_pretrained(m).cuda()
        return self.models[m]

    def predict(self,
                model: str = Input(description="Model to use for prediction", choices=MODELS, default=MODELS[0]),
                start_time: float = Input(description="Start time for the generation", default=0),
                end_time: float = Input(description="End time for the generation", default=10),
                top_p: float = Input(description="Nucleus sampling probability", default=0.95),
                mode: str = Input(description="Generation mode: 'AR' or 'AAR'", default='AR'),
                inputs: str = Input(description="Input tokens as string", default=""),
                controls: str = Input(description="Control tokens as string", default=""),
                debug: bool = Input(description="Enable debug outputs", default=False)
                ) -> str:
        """Run a single prediction on the model"""
        # Convert string inputs back to lists (assuming simple comma-separated tokens)
        model = self._load_model(m)

        if inputs and inputs != "":
            inputs = json.loads(inputs)
        else:
            inputs = None

        if controls and controls != "":
            controls = json.loads(controls)
        else:
            controls = None

        # Choose generation mode
        if mode == 'AR':
            result = generate_ar(model, start_time, end_time, input_tokens, control_tokens, top_p, debug)
        elif mode == 'AAR':
            result = generate(model, start_time, end_time, input_tokens, control_tokens, top_p, debug)
        else:
            raise ValueError("Invalid mode specified. Choose 'AR' or 'AAR'.")

        return result_str


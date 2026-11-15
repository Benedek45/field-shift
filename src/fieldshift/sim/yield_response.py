"""Pluggable relative-yield interface and a batched deployment adapter."""

from pathlib import Path

import torch

from fieldshift.sim.components import accelerator, fao_relative_yield


class YieldResponse:
    def __init__(self, edited_weather=False):
        self.net = None
        self.columns = []
        self.reason = "Edited weather: bucket-driven FAO-33 is mandatory." if edited_weather else ""
        self.path = "FAO-33"
        if edited_weather:
            return
        # Catch only deployment loading failures; inference failures must surface.
        try:
            from fieldshift.model.expanded import predict_calibrated_relative_yield
            from fieldshift.model.probabilistic import load_response

            path = Path("data/processed/models/yield_response_v2_power.pt")
            self.net, self.columns, self.metadata = load_response(path)
            self.single_predictor = predict_calibrated_relative_yield
            self.path = "Calibrated NASA POWER stage MLP"
            self.reason = "Saved model loaded; batched adapter uses the deployment function's identical lognormal moments."
        except (ImportError, FileNotFoundError, OSError, RuntimeError, KeyError, ValueError) as error:
            self.reason = f"Saved model could not load ({type(error).__name__}: {error}); using FAO-33."

    def relative_yield(self, stage_features, traits, soil, irrigated_share):
        if self.net is None:
            return fao_relative_yield(stage_features, traits, soil, irrigated_share)
        return self.single_predictor(stage_features, traits, soil, irrigated_share)

    def batch_matrix(self, x):
        """Feature matrix in ``self.columns`` order -> lognormal mean and SD of relative yield."""
        with torch.inference_mode():
            mean, sd = self.net.distribution(x)
            variance = sd.square()
            ratio = (mean + variance / 2).exp()
            spread = ratio * variance.expm1().sqrt()
        return ratio, spread

    def batch(self, records):
        if self.net is None:
            raise ValueError("FAO-33 requires daily bucket ET, not a static feature row")
        missing = set(self.columns) - records[0].keys()
        if missing:
            raise ValueError(f"Missing deployment features: {sorted(missing)}")
        x = torch.tensor([[r[k] for k in self.columns] for r in records],
                         dtype=torch.float32, device=accelerator())
        with torch.inference_mode():
            mean, sd = self.net.distribution(x)
            variance = sd.square()
            ratio = (mean + variance / 2).exp()
            spread = (mean + variance / 2) .exp() * variance.expm1().sqrt()
        return ratio, spread

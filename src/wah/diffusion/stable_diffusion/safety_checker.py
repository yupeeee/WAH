# https://github.com/woctezuma/stable-diffusion-safety-checker
import numpy as np
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from ...misc.typing import Device, Image, List, Tensor, Tuple

__all__ = [
    "SafetyChecker",
]


def cosine_distance(
    image_embeds: Tensor,
    text_embeds: Tensor,
) -> Tensor:
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def detect_bad_concepts(
    model: StableDiffusionSafetyChecker,
    clip_input: Tensor,
) -> Tuple[List[bool], Tensor]:
    # Get image embeddings
    pooled_output = model.vision_model(clip_input)[1]
    image_embeds = model.visual_projection(pooled_output)

    # Calculate cosine distances
    cos_dist = cosine_distance(image_embeds, model.concept_embeds)
    special_cos_dist = cosine_distance(image_embeds, model.special_care_embeds)

    # Get thresholds
    concept_thresholds = model.concept_embeds_weights
    special_thresholds = model.special_care_embeds_weights

    # Calculate scores
    concept_scores = cos_dist - concept_thresholds.unsqueeze(0)
    special_scores = special_cos_dist - special_thresholds.unsqueeze(0)
    # Add 0.01 to concept scores if any special scores are positive
    has_special = torch.any(special_scores > 0, dim=1)
    concept_scores = torch.where(
        has_special.unsqueeze(1),
        concept_scores + 0.01,
        concept_scores
    )

    # Detect bad concepts
    has_nsfw = torch.any(concept_scores > 0, dim=1).tolist()

    return has_nsfw, concept_scores


class SafetyChecker:
    MODEL_ID = "CompVis/stable-diffusion-safety-checker"

    def __init__(self, device: Device = "cpu") -> None:
        self.device = device
        self.processor = CLIPImageProcessor()
        self.model = StableDiffusionSafetyChecker.from_pretrained(
            self.MODEL_ID
        ).to(self.device)

    def to(self, device: Device) -> "SafetyChecker":
        self.device = device
        self.model = self.model.to(self.device)
        return self

    def __call__(self, images: List[Image]):
        clip_input = torch.tensor(
            np.array(self.processor(images).pixel_values)).to(self.device)
        has_nsfw, concept_scores = detect_bad_concepts(
            self.model, clip_input)
        return has_nsfw, concept_scores

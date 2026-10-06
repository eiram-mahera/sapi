from .base import BaseExperiment, ExperimentConfig, ExperimentResult
from .inference import InferenceExperiment
from .fewshot_finetuning import FewShotFinetuningExperiment

EXPERIMENT_REGISTRY = {
    "inference":          InferenceExperiment,
    "fewshot_finetuning": FewShotFinetuningExperiment,
}


def get_experiment(experiment_type: str) -> type:
    if experiment_type not in EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Unknown experiment type '{experiment_type}'. "
            f"Choose from: {list(EXPERIMENT_REGISTRY.keys())}"
        )
    return EXPERIMENT_REGISTRY[experiment_type]

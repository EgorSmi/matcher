import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple


LOG = logging.getLogger(__name__)


@dataclass
class CatboostConfig:
    task_type: str
    devices: str = "0"

    custom_metric: str = "PRAUC"
    eval_metric: str = "PRAUC"
    loss_function: str = "Logloss"
    iterations: int = 100
    learning_rate: float = 0.01
    l2_leaf_reg: str = 100.0
    bootstrap_type: str = "Bernoulli"
    boosting_type: str = "Plain"
    verbose: bool = True
    random_strength: float = 1000.0
    subsample: float = 0.98
    sampling_unit: str = None
    depth: int = 2
    grow_policy: str = "SymmetricTree"
    metric_period: int = 10
    random_seed: int = 42
    border_count: float = 1024

    golden_feature: List[str] = ("matching_bert_score", )

    def catboost_params(self, feature_names=None):
        result = asdict(self)
        if self.task_type == "CPU":
            del result["devices"]
        non_catboost_options = [
            "golden_feature",
        ]
        for option in non_catboost_options:
            del result[option]
        if self.golden_feature is not None:
            if feature_names is not None:
                quantization = []
                for feature_name in self.golden_feature:
                    if feature_name is not None:
                        try:
                            quantization.append(f"{feature_names.index(feature_name)}:border_count=1024")
                        except ValueError:
                            LOG.warning(f"Golden feature {feature_name} is absent in dataset!")
                    else:
                        LOG.info("None feature in list of golden features")
                if len(quantization) != 0:
                    result["per_float_feature_quantization"] = quantization
                else:
                    LOG.warning(f"Golden feature list is empty!")
            else:
                LOG.warning(f"Golden feature {self.golden_feature} is ignored because feature names are not given")

        result = {k: v for k, v in result.items() if v is not None}

        return result

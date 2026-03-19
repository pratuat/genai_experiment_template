"""Mlflow utility functions."""

import os
import yaml
from langchain_openai import ChatOpenAI


def load_model(model_name: str) -> ChatOpenAI:
    """Load a model from MLflow AI Gateway by model name."""
    mlflow_host = os.environ.get("MLFLOW_AI_GATEWAY_URI")

    llm = ChatOpenAI(
        model=model_name,
        base_url=f"{mlflow_host}/gateway/mlflow/v1",
        api_key="",
    )

    return llm


def load_models(groups: list[str]) -> list[ChatOpenAI]:
    """Load models from MLflow AI Gateway by list of model groups."""
    existing_groups = yaml.load(open("llms.yaml", "r"), Loader=yaml.FullLoader)[
        "groups"
    ]

    if groups and all(group in existing_groups.keys() for group in groups):
        models = []

        for group in groups:
            for el in existing_groups[group]:
                el["model"] = load_model(el["model_name"])
                models.append(el)

        return models
    else:
        raise ValueError(f"Invalid model groups: {groups}.")

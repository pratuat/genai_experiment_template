"""Mlflow utility functions."""

import os
import yaml
import pandas as pd
from langchain_openai import ChatOpenAI
import mlflow


def load_model(model_name: str) -> ChatOpenAI:
    """Load a model from MLflow AI Gateway by model name."""
    mlflow_host = os.environ.get("MLFLOW_AI_GATEWAY_URI")

    llm = ChatOpenAI(
        model=model_name,
        base_url=f"{mlflow_host}/gateway/mlflow/v1",
        api_key="",
    )

    return llm


def load_models_by_groups(groups: list[str]) -> list[ChatOpenAI]:
    """Load models from MLflow AI Gateway by list of model groups."""
    existing_groups = yaml.load(open("llms.yaml", "r"), Loader=yaml.FullLoader)[
        "groups"
    ]

    if groups and all(group in existing_groups.keys() for group in groups):
        models = []

        for group in groups:
            for el in existing_groups[group]:
                el["model"] = load_model(el["name"])
                models.append(el)

        return models
    else:
        raise ValueError(f"Invalid model groups: {groups}.")


def df_to_mlflow_records(
    df: pd.DataFrame, inputs_cols: list[str], expectations_cols: list[str]
) -> mlflow.data.Dataset:
    records = [
        {
            "inputs": row[inputs_cols].to_dict(),
            "expectations": row[expectations_cols].to_dict(),
        }
        for _, row in df.iterrows()
    ]
    return records


def format_evaluation_results(result_df):
    def _get_assesment_value(assessment):
        if assessment.get('feedback'):
            # case: assessment created by a scorer
            return assessment.get('feedback', {}).get('value')
        elif assessment.get('expectation'):
            # case: origin expectation from dataset
            return assessment.get('expectation', {}).get('value')

    def _get_row(row):
        record_dict = {}
        try:
            inputs = {f"inputs.{k}": v for k, v in row['request']['inputs'].items()} if row.get('request', {}).get('inputs') else {}
            record_dict.update(inputs)
            outputs = {f"outputs.{k}": v for k, v in row['response']['parsed_response'].items()} if row.get('response', {}).get('parsed_response') else {}
            record_dict.update(outputs)
            assessments = {f"assessments.{assessment['assessment_name']}": _get_assesment_value(assessment) for assessment in row['assessments']} if row.get('assessments') else {}
            record_dict.update(assessments)

            return record_dict
        except:
            return record_dict
    
    return pd.json_normalize(result_df.apply(_get_row, axis=1))
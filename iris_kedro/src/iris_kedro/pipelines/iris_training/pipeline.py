from functools import partial

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_data,
    save_final_assets,
    select_best_model,
    train_single_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Data Preparation
            node(
                func=prepare_data,
                inputs=["params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="prepare_data_node",
            ),
            # Train Models
            node(
                func=partial(train_single_model, model_name="RandomForest"),
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:model_options.RandomForest",
                ],
                outputs="eval_rf",
                name="train_rf_node",
            ),
            node(
                func=partial(train_single_model, model_name="LogisticRegression"),
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:model_options.LogisticRegression",
                ],
                outputs="eval_lr",
                name="train_lr_node",
            ),
            node(
                func=partial(train_single_model, model_name="SVM"),
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:model_options.SVM",
                ],
                outputs="eval_svm",
                name="train_svm_node",
            ),
            node(
                func=partial(train_single_model, model_name="KNN"),
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:model_options.KNN",
                ],
                outputs="eval_knn",
                name="train_knn_node",
            ),
            # Select Best Model
            node(
                func=select_best_model,
                inputs=["eval_rf", "eval_lr", "eval_svm", "eval_knn"],
                outputs="best_eval",
                name="select_best_model_node",
            ),
            # Save final assets
            node(
                func=save_final_assets,
                inputs="best_eval",
                outputs=None,
                name="save_final_assets_node",
            ),
        ]
    )

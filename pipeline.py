"""
ML Pipeline Orchestrator
Coordinates all agents in the correct order for a complete ML pipeline.
"""

import pandas as pd
import numpy as np
import os
import json
import time
from typing import Dict, Any, Optional, Callable

# Import all agents
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_understanding_agent import DataUnderstandingAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.small_dataset_agent import SmallDatasetOptimizationAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.visualization_agent import VisualizationAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.research_paper_agent import ResearchPaperAgent

import warnings
warnings.filterwarnings('ignore')


class MLPipeline:
    """Orchestrates the complete ML pipeline using modular agents."""

    def __init__(
        self,
        output_dir: str = "/tmp/ml_toolbox_outputs",
        progress_callback: Callable = None,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.progress_callback = progress_callback or (lambda msg, pct: None)

        # Initialize all agents
        self.agents = {
            "data_understanding": DataUnderstandingAgent(),
            "preprocessing": PreprocessingAgent(),
            "feature_engineering": FeatureEngineeringAgent(),
            "small_dataset": SmallDatasetOptimizationAgent(),
            "model_training": ModelTrainingAgent(),
            "hyperparameter": HyperparameterOptimizationAgent(),
            "evaluation": EvaluationAgent(),
            "visualization": VisualizationAgent(output_dir=os.path.join(output_dir, "plots")),
            "explainability": ExplainabilityAgent(),
            "research_paper": ResearchPaperAgent(),
        }

        self.pipeline_results = {}
        self.start_time = None

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        dataset_name: str = "Dataset",
        task_type: str = "classification",
        top_k_features: int = None,
        hp_method: str = "random",
        hp_n_iter: int = 20,
        preprocessing_config: Dict = None,
        run_explainability: bool = True,
        run_hp_optimization: bool = True,
    ) -> Dict[str, Any]:
        """Execute the complete ML pipeline."""
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  ML TOOLBOX PIPELINE STARTING")
        print(f"  Dataset: {dataset_name} | Target: {target_col}")
        print(f"  Task: {task_type} | Samples: {len(df)}")
        print(f"{'='*60}\n")

        results = {
            "dataset_name": dataset_name,
            "target_col": target_col,
            "task_type": task_type,
            "pipeline_config": {
                "hp_method": hp_method,
                "hp_n_iter": hp_n_iter,
                "top_k_features": top_k_features,
            },
        }

        try:
            # ── Step 1: Data Understanding ──────────────────────────────
            self.progress_callback("🔍 Data Understanding Agent running...", 5)
            data_report = self.agents["data_understanding"].run(df, target_col)
            results["data_understanding"] = data_report
            print()

            # ── Step 2: Preprocessing ────────────────────────────────────
            self.progress_callback("🔧 Preprocessing Agent running...", 15)
            X, y, prep_report = self.agents["preprocessing"].run(
                df, target_col, config=preprocessing_config
            )
            results["preprocessing"] = prep_report
            print()

            # ── Step 3: Small Dataset Optimization ───────────────────────
            self.progress_callback("📦 Small Dataset Optimization running...", 25)
            small_report = self.agents["small_dataset"].run(X, y, task_type)
            results["small_dataset"] = small_report
            print()

            # ── Step 4: Feature Engineering ──────────────────────────────
            self.progress_callback("⚙️ Feature Engineering Agent running...", 35)
            X_engineered, feat_report = self.agents["feature_engineering"].run(
                X, y, task_type=task_type, top_k=top_k_features
            )
            results["feature_engineering"] = feat_report
            print()

            # ── Step 5: Model Training ────────────────────────────────────
            self.progress_callback("🤖 Model Training Agent running...", 50)
            trained_models, train_report = self.agents["model_training"].run(
                X_engineered, y, task_type=task_type, n_samples=len(X_engineered)
            )
            results["model_training"] = train_report
            print()

            # ── Step 6: Hyperparameter Optimization ──────────────────────
            if run_hp_optimization:
                self.progress_callback(f"🔬 Hyperparameter Optimization ({hp_method})...", 62)
                optimized_models, hp_report = self.agents["hyperparameter"].run(
                    X_engineered, y, trained_models, task_type, method=hp_method, n_iter=hp_n_iter
                )
                results["hyperparameter"] = hp_report
                models_to_eval = optimized_models
                print()
            else:
                results["hyperparameter"] = {"status": "skipped"}
                models_to_eval = trained_models

            # ── Step 7: Evaluation ───────────────────────────────────────
            self.progress_callback("📊 Evaluation Agent running...", 72)
            eval_report = self.agents["evaluation"].run(
                models_to_eval, X_engineered, y, task_type
            )
            results["evaluation"] = eval_report
            print()

            # ── Step 8: Visualization ─────────────────────────────────────
            self.progress_callback("📈 Visualization Agent generating plots...", 80)
            viz_report = self.agents["visualization"].run(
                df_original=df,
                X=X_engineered,
                y=y,
                models=models_to_eval,
                eval_report=eval_report,
                feature_report=feat_report,
                target_col=target_col,
            )
            results["visualization"] = viz_report
            print()

            # ── Step 9: Explainability ────────────────────────────────────
            if run_explainability:
                self.progress_callback("🧠 Explainability Agent running (SHAP + LIME)...", 88)
                try:
                    expl_report = self.agents["explainability"].run(
                        models=models_to_eval,
                        X=X_engineered,
                        y=y,
                        best_model_name=eval_report.get("best_model"),
                        feature_names=X_engineered.columns.tolist(),
                    )
                    results["explainability"] = expl_report
                except Exception as e:
                    results["explainability"] = {"status": "failed", "error": str(e)}
                print()
            else:
                results["explainability"] = {"status": "skipped"}

            # ── Step 10: Research Paper ───────────────────────────────────
            self.progress_callback("📝 Research Paper Agent generating report...", 94)
            paper_report = self.agents["research_paper"].run(
                dataset_name=dataset_name,
                data_report=data_report,
                preprocessing_report=prep_report,
                feature_report=feat_report,
                training_report=train_report,
                eval_report=eval_report,
                optimization_report=results.get("hyperparameter"),
                explainability_report=results.get("explainability"),
                small_dataset_report=small_report,
            )
            results["research_paper"] = paper_report
            print()

            # ── Summary ───────────────────────────────────────────────────
            elapsed = round(time.time() - self.start_time, 2)
            results["pipeline_summary"] = {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "best_model": eval_report.get("best_model"),
                "best_score": eval_report.get("ranking", [{}])[0].get("score") if eval_report.get("ranking") else None,
                "n_models_trained": len(trained_models),
                "n_plots_generated": len(self.agents["visualization"].get_all_figures()),
            }

            self.pipeline_results = results
            self.progress_callback("✅ Pipeline complete!", 100)

            print(f"\n{'='*60}")
            print(f"  PIPELINE COMPLETE in {elapsed}s")
            print(f"  Best Model: {eval_report.get('best_model')}")
            print(f"  Best Score: {results['pipeline_summary']['best_score']}")
            print(f"{'='*60}\n")

            return results

        except Exception as e:
            error_msg = f"Pipeline failed at step: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            results["pipeline_summary"] = {"status": "failed", "error": error_msg}
            self.progress_callback(f"❌ Error: {str(e)[:100]}", 0)
            return results

    def get_visualization_agent(self) -> VisualizationAgent:
        return self.agents["visualization"]

    def get_explainability_agent(self) -> ExplainabilityAgent:
        return self.agents["explainability"]

    def get_research_paper_text(self) -> str:
        if "research_paper" in self.pipeline_results:
            return self.agents["research_paper"].get_full_paper_text()
        return "Research paper not generated yet."

    def get_model_comparison_df(self) -> pd.DataFrame:
        return self.agents["evaluation"].get_comparison_df()

    def save_results(self) -> str:
        """Save results summary to JSON."""
        # Make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            return obj

        # Exclude non-serializable components
        save_results = {}
        skip_keys = {'visualization', 'best_estimator'}
        for k, v in self.pipeline_results.items():
            if k not in skip_keys:
                save_results[k] = make_serializable(v)

        path = os.path.join(self.output_dir, "pipeline_results.json")
        with open(path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)

        return path

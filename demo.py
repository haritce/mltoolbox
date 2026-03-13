"""
Demo script — runs the ML pipeline on the Iris dataset programmatically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import MLPipeline
from data.example_datasets import load_example


def main():
    print("\n🤖 ML TOOLBOX — DEMO RUN")
    print("=" * 50)

    # Load example dataset
    df, target_col = load_example("🌸 Iris (150 samples, multiclass)")
    print(f"Dataset: Iris | Shape: {df.shape} | Target: {target_col}")

    # Initialize and run pipeline
    pipeline = MLPipeline(output_dir="/tmp/ml_toolbox_demo")

    results = pipeline.run(
        df=df,
        target_col=target_col,
        dataset_name="Iris",
        task_type="classification",
        top_k_features=4,
        hp_method="random",
        hp_n_iter=10,
        run_explainability=True,
        run_hp_optimization=True,
    )

    # Print summary
    summary = results.get("pipeline_summary", {})
    print(f"\n✅ STATUS: {summary.get('status', 'unknown')}")
    print(f"⏱️  Time: {summary.get('elapsed_seconds')}s")
    print(f"🏆 Best Model: {summary.get('best_model')}")
    print(f"📊 Best Score: {summary.get('best_score')}")
    print(f"📈 Plots: {summary.get('n_plots_generated')}")

    # Print model comparison
    print("\n📊 Model Comparison:")
    comparison_df = pipeline.get_model_comparison_df()
    if not comparison_df.empty:
        print(comparison_df[['Model', 'accuracy', 'f1_score', 'roc_auc']].to_string(index=False))

    # Save results
    json_path = pipeline.save_results()
    print(f"\n💾 Results saved to: {json_path}")

    print("\n🎉 Demo complete! Run `streamlit run app.py` to open the web interface.")


if __name__ == "__main__":
    main()

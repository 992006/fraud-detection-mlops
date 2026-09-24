import mlflow
from mlflow.tracking import MlflowClient

def main():
    client = MlflowClient()
    experiment_name = "fraud-detection-baseline"
    
    # 1. Find our experiment
    experiment = client.get_experiment_by_name(experiment_name)
    
    # 2. Search for the best run based on the highest F1 score
    print("Searching for the best performing model...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name = best_run.data.params["model_name"]
    f1_score = best_run.data.metrics["f1_score"]
    
    print(f"🏆 Best model: {model_name} with F1-Score: {f1_score:.4f}")
    print(f"Run ID: {run_id}")
    
    # 3. Register the model in the MLflow Model Registry
    # The artifact path we used in training was f"{model_name}_model"
    model_uri = f"runs:/{run_id}/{model_name}_model"
    registry_name = "FraudDetectionModel"
    
    print(f"\nRegistering model as '{registry_name}'...")
    mv = mlflow.register_model(model_uri, registry_name)
    
    # 4. Promote the model to "Production" using MLflow Aliases
    print("Promoting model to Production...")
    client.set_registered_model_alias(
        name=registry_name,
        alias="Production",
        version=mv.version
    )
    
    print(f"\n✅ SUCCESS! '{registry_name}' version {mv.version} is now in Production!")
    print("👉 Open your MLflow UI and click the 'Models' tab to see it.")

if __name__ == "__main__":
    main()
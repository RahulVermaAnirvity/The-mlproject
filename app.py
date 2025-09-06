import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer

import mlflow
import mlflow.sklearn

def main():
    logging.info("Execution has started.")

    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingested. Train path: {train_data_path}, Test path: {test_data_path}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transormation(train_data_path, test_data_path)
        logging.info("Data transformation completed.")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed. R2 Score: {r2}")

        # Step 4: Logging with MLflow
        mlflow.set_experiment("DecisionTree_Experiment")
        with mlflow.start_run():
            # If you want to log the model, we need the model object.
            # Assuming ModelTrainer has `model` attribute
            if hasattr(model_trainer, 'model'):
                mlflow.sklearn.log_model(model_trainer.model, "decision_tree_model")
            mlflow.log_metric("r2_score", r2)
        logging.info("Model and metrics logged to MLflow.")

        print("Model training and MLflow logging completed successfully.")

    except Exception as e:
        logging.error("An error occurred during execution.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()

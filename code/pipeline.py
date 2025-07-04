import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile

role = sagemaker.get_execution_role()
region = sagemaker.Session().boto_region_name
bucket_name = 'gsir-ey-test-123'
prefix = 'data'

pipeline_session = PipelineSession()

prefix = 'data'
bucket = bucket_name = 'gsir-ey-test-123'

script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.0-1"),
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="split-data",
    role=role,
    sagemaker_session=pipeline_session
)

processing_step = ProcessingStep(
    name="SplitData",
    processor=script_processor,
    inputs=[
        ProcessingInput(
            source=f's3://{bucket}/{prefix}/fraud_data.csv',  # your raw data file
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ],
    code="split_data.py",  # path to your processing script
    job_arguments=[
        "--input-data", "/opt/ml/processing/input/fraud_data.csv",
        "--train-output", "/opt/ml/processing/train",
        "--test-output", "/opt/ml/processing/test",
        "--label-column", "is_fraud"
    ]
)


from sagemaker.inputs import TrainingInput

xgb_image_uri = sagemaker.image_uris.retrieve("xgboost", region, "1.5-1")

xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=xgb_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size=5,
    max_run=300,
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=pipeline_session
)

xgb_estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=10,
    max_depth=2,
    eta=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc"
)

training_step = TrainingStep(
    name="TrainXGBoostModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)


pipeline = Pipeline(
    name="FraudDetectionXGBoostPipeline",
    steps=[processing_step, training_step],
    sagemaker_session=pipeline_session,
)

pipeline.upsert(role_arn=role)
print("Pipeline created!")

execution = pipeline.start()
execution.wait()
print("Pipeline execution completed!")

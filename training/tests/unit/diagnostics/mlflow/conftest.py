# Skip Azure ML logger test unless the optional [azure] extras are installed.
# The test requires `azure-ai-ml` + `azure-identity`, available via
# `pip install anemoi-training[azure]`. Without them the file fails at import
# and aborts the whole training/tests/unit collection.
collect_ignore = ["test_azureml_mlflow_logger.py"]

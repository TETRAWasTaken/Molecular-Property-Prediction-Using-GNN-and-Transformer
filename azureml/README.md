# Azure ML Training Job

This folder contains a command job definition for training the transformer pipeline on Azure Machine Learning.

## Files

- `transformer-train-job.yml`: Azure ML command job spec that runs `Transformers_2/manual_run.py`.
- `transformer-conda.yml`: Conda environment used by the training job.
- `hybrid-train-job.yml`: Azure ML command job spec that runs `main.py` (GIN + ChemBERTa hybrid model) on serverless compute.
- `hybrid-conda.yml`: Conda environment used by the hybrid training job.

## Submit

From the repository root:

```bash
az ml job create -f azureml/transformer-train-job.yml
```

For hybrid training:

```bash
az ml job create -f azureml/hybrid-train-job.yml
```

## Notes

- The job uses Azure ML serverless compute, so you do not need to create or manage a persistent compute instance.
- `resources.instance_type` selects the VM size used for the run. Change it if your workspace quota does not allow `Standard_NC4as_T4_v3`.
- The job uses `Dataset/New_QM9/molecule_properties.csv` by default.
- Model checkpoints and preprocessing cache are written to Azure ML outputs so they are uploaded after the run.
- The hybrid job injects dataset and output paths via environment variables (`MOLECULE_CSV_PATH`, `ATOM_CSV_PATH`, `TOKENIZED_CACHE_PATH`, `HYBRID_MODEL_OUTPUT_PATH`) so local defaults still work outside Azure ML.

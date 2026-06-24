# Azure ML Training — Setup & Usage Guide (CPU Serverless)

This folder contains Azure ML command-job definitions and environment configurations for training the molecular property prediction models in the cloud on **CPU-only serverless compute**.

---

## Files

| File | Purpose |
|------|---------|
| `hybrid-train-job.yml` | Runs `main.py` — trains the full **GIN + ChemBERTa hybrid** model |
| `hybrid-conda.yml` | Conda environment for the hybrid job (CPU PyTorch + CPU PyG stack) |
| `transformer-train-job.yml` | Runs `Transformers_2/manual_run.py` — fine-tunes standalone **ChemBERTa** model |
| `transformer-conda.yml` | Conda environment for standalone ChemBERTa training |
| `gin-train-job.yml` | Runs `GIN_2/manual_run.py` — trains standalone **GIN** model |
| `gin-conda.yml` | Conda environment for standalone GIN training (CPU PyTorch + CPU PyG stack) |

---

## Prerequisites

### 1. Install the Azure ML CLI (v2)

```bash
pip install azure-ai-ml
az extension add --name ml --upgrade
az login
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
```

### 2. Set your workspace defaults

```bash
az configure --defaults \
  group=<YOUR_RESOURCE_GROUP> \
  workspace=<YOUR_WORKSPACE_NAME> \
  location=<REGION>   # e.g. eastus
```

Verify the setup:
```bash
az ml workspace show
```

---

## Step 1 — Upload the QM9 Dataset to your Datastore

The job YAMLs reference:
```
azureml://datastores/workspaceblobstore/paths/qm9/molecule_properties.csv
azureml://datastores/workspaceblobstore/paths/qm9/atom_properties.csv
```

Upload both datasets using the CLI (run from the project root):
```bash
az ml data create \
  --name qm9-molecule-csv \
  --version 1 \
  --type uri_file \
  --path Dataset/New_QM9/molecule_properties.csv \
  --datastore workspaceblobstore

az ml data create \
  --name qm9-atom-csv \
  --version 1 \
  --type uri_file \
  --path Dataset/New_QM9/atom_properties.csv \
  --datastore workspaceblobstore
```

---

## Step 2 — (Optional) Cache ChemBERTa Weights

The hybrid and transformer jobs download `seyonec/ChemBERTa-zinc-base-v1` from the Hugging Face Hub on first run (`HF_HUB_OFFLINE: "0"`). To avoid repeated downloads across runs, upload cached weights and mount them:

```bash
# On your local machine, the cache is at ~/.cache/huggingface/hub/
az ml data create \
  --name chemberta-weights \
  --type uri_folder \
  --path ~/.cache/huggingface/hub/models--seyonec--ChemBERTa-zinc-base-v1 \
  --datastore workspaceblobstore
```

Then add to the input section of `hybrid-train-job.yml` or `transformer-train-job.yml`:
```yaml
  hf_cache:
    type: uri_folder
    path: azureml:chemberta-weights@latest
```
And add to `environment_variables`:
```yaml
  TRANSFORMERS_CACHE: ${{inputs.hf_cache}}
  HF_HUB_OFFLINE: "1"
```

---

## Step 3 — Submit Jobs

Submit jobs from the **repository root** (not from within the `azureml/` folder):

### A. Hybrid Model (GIN + ChemBERTa)
```bash
az ml job create \
  --file azureml/hybrid-train-job.yml \
  --stream   # Stream logs to the terminal; remove to submit asynchronously
```

### B. Standalone ChemBERTa Transformer
```bash
az ml job create \
  --file azureml/transformer-train-job.yml \
  --stream
```

### C. Standalone GIN Model
```bash
az ml job create \
  --file azureml/gin-train-job.yml \
  --stream
```

---

## Step 4 — Monitor Jobs

```bash
# List recent jobs
az ml job list --query "[?status=='Running']" -o table

# Stream logs from a specific run
az ml job stream --name <RUN_NAME>

# Open in Azure ML Studio
az ml job show --name <RUN_NAME> --query services.Studio.endpoint -o tsv
```

---

## Step 5 — Download Outputs

After a run completes, download checkpoints and outputs to your local directory:
```bash
az ml job download \
  --name <RUN_NAME> \
  --output-name model_dir \
  --download-path ./models
```

---

## Serverless CPU Compute Configuration

These configuration files utilize **Azure ML Serverless Compute** (by omitting the `compute:` targeting field, Azure ML automatically provisions the instance on submission and tears it down on completion). This is highly cost-effective as you only pay for compute active during the job run.

### Recommended CPU Instance Types (SKUs)

| SKU | vCPUs | RAM (GB) | Recommended Use |
|-----|-------|----------|-----------------|
| `Standard_D4as_v4` | 4 | 16 | Small preprocessing and testing |
| `Standard_D8as_v4` | 8 | 32 | Standalone GIN or ChemBERTa training |
| `Standard_D16as_v4` | 16 | 64 | Hybrid training / large batch preprocessing |
| `Standard_E16s_v3` | 16 | 128 | Memory-heavy training runs |

Specify the CPU instance in the `resources` block of any job YAML file:
```yaml
resources:
  instance_type: Standard_D16as_v4
  instance_count: 1
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Environment resolution failed` | Ensure the conda file references the CPU PyTorch/PyG repositories exactly. |
| `ModuleNotFoundError: torch_geometric` | Make sure you are using `hybrid-conda.yml` or `gin-conda.yml` which include PyG CPU wheels, rather than `transformer-conda.yml`. |
| `FileNotFoundError: molecule_properties.csv` | Verify your dataset paths or upload name using `az ml datastore show --name workspaceblobstore`. |
| `interactive input required` | Ensure you run GIN with `--mol_csv` and `--atom_csv` arguments (like in `gin-train-job.yml`) to prevent interactive prompts. |

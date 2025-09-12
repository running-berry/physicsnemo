# Guide on running Corrdiff training and sample generation with CWA dataset

You can download CWA dataset from [CWA_DATASET](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cwa)

Noted: the total size is approximately 1.5–2 TB, so be sure you have sufficient storage before starting the download.

---

## Table of Contents

1. [Build the Docker Image](#build-the-docker-image)  
2. [Run the Container](#run-the-container)  
3. [Train the **Regression** Model](#train-the-regression-model)  
4. [Train the **Diffusion** Model](#train-the-diffusion-model)  
5. [Generate Samples](#generate-samples)  

---

## 1. Build the Docker Image

Build the **physicsnemo** image from the `deploy` stage:

```bash
docker build -t physicsnemo:deploy \
  --build-arg TARGETPLATFORM=linux/amd64 \
  --target deploy \
  -f Dockerfile .
```

- **Tag**: `physicsnemo:deploy`  
- **Target platform**: `linux/amd64`  
- **Build stage**: `deploy`

---

## 2. Run the Container

Replace `[/path/to/physicsnemo]` with your local repo path:

```bash
docker run \
  --gpus '"device=0,1"' \
  --shm-size=32g \
  --ulimit stack=67108864 \
  --rm -it \
  --name wrf_container \
  -v [/path/to/physicsnemo]:/workspace/physicsnemo \
  -v [/path/to/cwa_dataset]:/workspace/cwa_dataset \
  physicsnemo:deploy \
  bash
```

- **GPUs**: `device=0,1`  
- **Shared memory**: `32g`  
- **Container name**: `wrf_container`  
- **Volumes**:
  - `…/physicsnemo` → `/workspace/physicsnemo`
  - `…/cwa_dataset` → `/workspace/cwa_dataset`

---

## 3. Train the **Regression** Model

1. **Reduce** `total_batch_size`  
    - Change total_batch_size in examples/weather/corrdiff/conf/base/training/base_all.yaml to a smaller number to avoid GPU out of memory promblem.
   Edit `examples/weather/corrdiff/conf/base/training/base_all.yaml`  
   ```yaml
   training:
     hp:
       total_batch_size: <smaller-number>
   ```
2. **Configure data path**  
 
   In `examples/weather/corrdiff/conf/config_training_taiwan_regression.yaml` and  
   `examples/weather/corrdiff/conf/base/dataset/cwb.yaml` set:
   ```yaml
   dataset:
     data_path: /workspace/cwa_dataset/cwa_dataset.zarr
   ```
3. **Run training**  
   ```bash
   cd examples/weather/corrdiff
   python train.py --config-name=config_training_taiwan_regression.yaml
   ```

---

## 4. Train the **Diffusion** Model

1. **Configure data path & regression checkpoint**  
   In `examples/weather/corrdiff/conf/config_training_taiwan_diffusion.yaml`:
    - change dataset[data_path] to the path of cwa_dataset.zarr in the docker container.
    - change regression_checkpoint_path to the regression model (UNet**.mdlus) checkpoint
   ```yaml
   dataset:
     data_path: /workspace/cwa_dataset/cwa_dataset.zarr

   training:
     io:
       regression_checkpoint_path: /workspace/physicsnemo/corrdiff/checkpoints_regression/UNet*.mdlus
   ```
2. **Run training**  
   ```bash
   cd examples/weather/corrdiff
   python train.py --config-name=config_training_taiwan_diffusion.yaml
   ```

---

## 5. Generate Samples

1. **Configure generation**  
   In `examples/weather/corrdiff/conf/config_generate_taiwan.yaml`:
      - change res_ckpt_filename to the diffusion model (EDM**.mdlus) checkpoint (can be found in corrdiff/checkpoints_diffusion)
      - change reg_ckpt_filename to the regression model (UNet**.mdlus) checkpoint (can be found in corrdiff/checkpoints_regression)
   ```yaml
   generation:
    io:
     res_ckpt_filename: /workspace/physicsnemo/corrdiff/checkpoints_diffusion/EDM*.mdlus
     reg_ckpt_filename: /workspace/physicsnemo/corrdiff/checkpoints_regression/UNet*.mdlus
   ```
2. **Run generation**  
   ```bash
   python generate.py --config-name=config_generate_taiwan.yaml
   ```

---


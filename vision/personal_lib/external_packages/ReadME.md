# Install External Package
If you want to use following packages, please type the command bellow.

## mamba ssm (for Vim)
docker/cuda12 image failed. So please use docker/cuda11.  
The reason might be because our docker/cuda12 uses nvcr and https://github.com/Dao-AILab/causal-conv1d/issues/10#issuecomment-1869087416.

```
# --user is needed not to create .so file in the mamba-1p1p1 directory. The .so file is harmful to the different GPUs if the home directory is shared among different GPU architectures.
pip install personal_lib/external_packages/mamba-1p1p1 --user 
```
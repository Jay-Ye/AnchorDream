pip install             \
    s3fs                \
    lovely_tensors      \
    wandb==0.16.6       \
    pygame              \
    pyopengl            \
    imageio             \
    h5py
pip install -U "moviepy<2"
apt-get update && apt-get install -y      \
        freeglut3-dev
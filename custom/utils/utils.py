import os
from imaginaire.utils import distributed
from typing import Optional
import shutil
import errno
from os.path import expandvars, expanduser
import subprocess
import sys
from typing import List
import torch
import torch.distributed as dist

def exec_s3_cp(
    src: str,
    dest: str,
    extras: List[str] = [],
    print_to_console: bool = True,
):
    cmd = ["aws", "s3", "cp", src, dest] + extras
    my_env = os.environ.copy()

    if print_to_console:
        print("====================================")
        print(f"exec:  {' '.join(cmd)}")
        print("====================================")

    os.makedirs(expanduser("~/tmp"), exist_ok=True)
    tmp_log = expanduser("~/tmp/s3_cp.log")
    with open(tmp_log, "wb") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=my_env)
        for c in iter(lambda: process.stdout.read(1), b""):
            if print_to_console:
                sys.stdout.buffer.write(c)
            f.write(c)

        # check aws s3 return code.
        process.communicate()
        assert process.returncode == 0, process.returncode

    with open(tmp_log, "rb") as f:
        outputs = f.readlines()
    # bytes to string, and string new line
    outputs = [line.decode().rstrip() for line in outputs]
    return cmd, outputs


def maybe_exec_s3_cp(src, dest, *args, **kwargs):
    """
    Same as exec_s3_cp(), but will return True if success, or will catch
    errors and return False.
    """
    try:
        exec_s3_cp(src, dest, *args, **kwargs)
        return True
    except Exception as e:
        print(f"Error copying from {repr(src)} to {repr(dest)}:\n{repr(e)}")
        return False

def download_from_s3(
    path: str = None,
    save_dir: str = "~/tmp",
    aws_profile: Optional[str] = "default",
) -> str:
    """
    Downloads a file from either an EFS or S3 path to a local save_dir.

    The input path may be specified either as a mounted file-path
    (e.g., '~/efs/...'), or as an S3 path (e.g., 's3://...').
    In both cases, the specified save_dir is used to download the file
    (if not already present). If an file-system type path is provided,
    a corresponding path in the S3_LBM_BUCKET is used to locate the
    referenced file and download it from S3 to the local save_dir.

    Args:
        path: The S3 path to the file to be downloaded
            (e.g., 's3://bucket_name/path/to/file').
        save_dir (str): The local directory where the file should be saved.
            Defaults to '~/tmp'.
        aws_profile (Optional[str]): The AWS CLI profile to use for
            authentication. Defaults to DEFAULT_AWS_PROFILE.

    Returns:
        str: The path to the downloaded file in the cache (save_dir).
    """
    if not save_dir.endswith("/"):
        save_dir = save_dir + "/"
    home_path = expandvars(expanduser("~"))
    save_path = ""
    s3_path = ""
    efs_path = ""

    if path.startswith("s3://"):
        # Native s3 path. Proceed directly cache / return.
        # sagemaker training checkpoint: s3://robotics-manip-lbm/sagemaker_outputs/main-single-task/main-bimanual-place-tape-int-training-2024-08-06-08-35-04/epoch_400-step_22456.ckpt # noqa
        s3_path = path
        save_path = path.replace("s3://", save_dir)

    save_local_path = expandvars(expanduser(save_path))

    if os.path.exists(save_local_path):
        # Already present in cache
        return save_local_path
    elif maybe_exec_s3_cp(
        s3_path,
        save_local_path,
        print_to_console=True,
    ):
        # downloaded from s3 successfully, so return
        return save_local_path
    else:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            f"{path}",
        )

def distributed_download_from_s3(
    path: str = None,
    save_dir: str = "~/tmp",
    aws_profile: Optional[str] = "default",
) -> str:
    """
    Wrapper around `download_from_s3` to ensure only rank 0 downloads the file.
    All other ranks wait for the download to complete and proceed afterward.
    """
    # Prepare the expanded local path (same logic as in download_from_s3)
    if not save_dir.endswith("/"):
        save_dir += "/"
    save_path = path.replace("s3://", save_dir)
    save_local_path = expandvars(expanduser(save_path))

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0

    if rank == 0:
        if not os.path.exists(save_local_path):
            print(f"Downloading {path} to {save_local_path} from rank {rank}")
            download_from_s3(path=path, save_dir=save_dir, aws_profile=aws_profile)
        else:
            print(f"File {path} already exists in {save_local_path} from rank {rank}, skipping download")
    if is_distributed:
        dist.barrier()

    return save_local_path

def get_data_folder():
    """
    Returns folder for pretrained models and checkpoints.
    Set ANCHORDREAM_DATA_ROOT environment variable to override the default.
    Default: current working directory (expects checkpoints/ to be a subdirectory).
    """
    return os.getenv('ANCHORDREAM_DATA_ROOT', '.')


def get_checkpoint(path):
    return path


def get_folder(path):
    return path

def get_resume(path):
    return path

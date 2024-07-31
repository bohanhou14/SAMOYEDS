from huggingface_hub import snapshot_download
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("model_name")
    args = parser.parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=f"models/{args.model_name}",
        local_dir_use_symlinks=True,
        force_download=True,
        resume_download=False
    )

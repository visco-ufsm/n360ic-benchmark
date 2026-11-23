import subprocess
from pathlib import Path


class LIC360Wrapper:
    def __init__(self, workdir: Path, script_path: Path):
        self.workdir = workdir
        (self.workdir / "out" / "reconstruction").mkdir(parents=True, exist_ok=True)
        (self.workdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.script_path = script_path

    def run(self, image_path: Path, model_idx: int, ssim: bool = False):
        enc_commands = [
            "python3",
            self.script_path.as_posix(),
            "--enc",
            "--img-list",
            image_path.as_posix(),
            "--code-list",
            image_path.stem,
            "--model-idx",
            str(model_idx),
        ]
        dec_commands = [
            "python3",
            self.script_path.as_posix(),
            "--dec",
            "--code-list",
            image_path.stem,
            "--out-list",
            (image_path.parent / f"dec_{image_path.name}"),
            "--model-idx",
            str(model_idx),
        ]
        if ssim:
            enc_commands.append("--ssim")
            dec_commands.append("--ssim")
        try:
            enc_result = subprocess.run(
                enc_commands,
                capture_output=True,
                check=True,
                text=True,
            )
            dec_result = subprocess.run(
                dec_commands,
                capture_output=True,
                check=True,
                text=True,
            )
            return enc_result.stdout, dec_result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

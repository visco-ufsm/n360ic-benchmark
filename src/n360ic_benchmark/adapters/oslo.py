import subprocess
from pathlib import Path

HEALPIX_RES = 4
PATCH_RES = 2
RATE_POINTS = [
    (0.0005, 1),
    (0.0009, 1),
    (0.0018, 1),
    (0.0035, 2),
    (0.0067, 3),
    (0.0130, 4),
    (0.0250, 5),
    (0.0483, 6),
    (0.0932, 7),
    (0.1800, 8),
]


class OSLOWrapper:
    def __init__(self, workdir: Path, gpu: bool = False):
        self.workdir = workdir
        (self.workdir / "healpix").mkdir(parents=True, exist_ok=True)
        (self.workdir / "neighbor_structure").mkdir(parents=True, exist_ok=True)
        (self.workdir / "out" / "reconstruction").mkdir(parents=True, exist_ok=True)
        (self.workdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def rate_points(self) -> list[str]:
        return RATE_POINTS

    def ERP_to_HEALPix(self, image: Path):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./generate_healpix_samples_and_split_to_train_and_test.py",
                    "-i",
                    image.parent.as_posix(),
                    "-e",
                    f"{image.name.split('.')[-1]}",
                    "-r",
                    f"{HEALPIX_RES}",
                    "-s",
                    "-o",
                    (self.workdir / "healpix").as_posix(),
                    "-t",
                    str(0.0),
                    "-v",
                    str(0.0),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

    def build_neighbor_struct(self):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./generate_and_save_healpix_oslo_struct.py",
                    "-hr",
                    f"{HEALPIX_RES}",
                    "-pr",
                    f"{PATCH_RES}",
                    "-o",
                    (self.workdir / "neighbor_structure").as_posix(),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

    def run(self, rate_point, checkpoint_file):
        lmbda, q = rate_point
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./main_sphere_compression.py",
                    "-m",
                    "SphereFactorizedPrior",
                    "-nd",
                    (self.workdir / "neighbor_structure").as_posix(),
                    "-o",
                    (self.workdir / "test").as_posix(),
                    "--lambda",
                    str(lmbda),
                    "-q",
                    str(q),
                    "--checkpoint-file",
                    checkpoint_file,
                    "--foldername-valtest",
                    (self.workdir / "out" / "reconstruction").as_posix(),
                    "-t",
                    (self.workdir / "healpix" / "test.txt"),
                    "-hr",
                    str(HEALPIX_RES),
                    "-prvt",
                    str(PATCH_RES),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)


class OSLOICWrapper(OSLOWrapper):
    def run(self, rate_point, checkpoint_file):
        lmbda, q = rate_point
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./main_sphere_compression.py",
                    "-m",
                    "SphereMeanScaleHyperprior",
                    "-nd",
                    (self.workdir / "neighbor_structure").as_posix(),
                    "-o",
                    (self.workdir / "test").as_posix(),
                    "--lambda",
                    str(lmbda),
                    "-q",
                    str(q),
                    "--checkpoint-file",
                    checkpoint_file,
                    "--foldername-valtest",
                    (self.workdir / "out" / "reconstruction").as_posix(),
                    "-t",
                    (self.workdir / "healpix" / "test.txt"),
                    "-hr",
                    str(HEALPIX_RES),
                    "-prvt",
                    str(PATCH_RES),
                    "--context-model",
                    "autoregressive",
                    "--attention",
                    "--nonlinearity",
                    "RB",
                    "--unpool-func",
                    "SDPAConvTranspose",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

import os

packages = [
    "pytorch torchvision torchaudio pytorch-cuda=11.8",
    "pytorch-lightning",
    "torchmetrics",
    "tensorboard",
    "grad-cam",
    "albumentations",
    "opencv",
    "matplotlib",
    "torchcam",
    "segmentation-models-pytorch",
    "transformers",
]

command = "conda install -y -c pytorch -c nvidia -c conda-forge " + " ".join(packages)
os.system(command)
os.system("pip install gradio")
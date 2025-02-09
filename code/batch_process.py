import os
import nbformat
from nbclient import NotebookClient

def execute_notebook_with_kernel_restart(notebook_path, output_path=None, timeout=0):
    """
    Executes a Jupyter notebook with kernel restart before running the cells.
    Clears all outputs after execution to reduce file size.
    Args:
        notebook_path (str): Path to the input notebook file.
        output_path (str): Path to save the executed notebook. If None, overwrites the original file.
        timeout (int): Execution timeout in seconds.
    """
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    # Clear all outputs
    if "vit" in notebook_path:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
    # Execute the notebook with kernel restart
    client = NotebookClient(nb, timeout=timeout, kernel_name='python3', restart_kernel=True)
    try:
        client.execute()
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}")

    # Clear all outputs
    if "vit" in notebook_path:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None

    # Save the cleared notebook
    if output_path is None:
        output_path = notebook_path  # Overwrite original
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Executed and cleared notebook saved to {output_path}")

notebooks = [
    "../notebooks/DenseNet131_gradual_unfreeze.ipynb",
    "../notebooks/DenseNet131_premult.ipynb",
    "../notebooks/DenseNet131_unfreeze.ipynb",
    "../notebooks/EfficientNet_gradual_unfreeze.ipynb",
    "../notebooks/EfficientNet_premult.ipynb",
    "../notebooks/EfficientNet_unfreeze.ipynb",
    "../notebooks/Resnet18_equalized.ipynb",
    "../notebooks/Resnet18_gradual_unfreeze.ipynb",
    "../notebooks/Resnet18_premult.ipynb",
    "../notebooks/Resnet18_raw.ipynb",
    "../notebooks/Resnet18_reordered.ipynb",
    "../notebooks/Resnet18_reordered_weighted.ipynb",
    "../notebooks/Resnet18_unfreeze.ipynb",
    "../notebooks/Resnet18_upscale.ipynb",
    "../notebooks/Resnet18_upscale_unfreeze.ipynb",
    "../notebooks/Resnet50_gradual_unfreeze.ipynb",
    "../notebooks/Resnet50_premult.ipynb",
    "../notebooks/Resnet50_unfreeze.ipynb",
    "../notebooks/vit-base-patch16-224.ipynb",
    "../notebooks/vit-large-patch16-384.ipynb",
    "../notebooks/vit-large-patch32-384.ipynb"
]


# Execute each notebook with kernel restart
for notebook in notebooks:
    execute_notebook_with_kernel_restart(notebook)
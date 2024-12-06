import os
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read, write
import nbformat

def batch_process_notebooks(notebook_list, output_dir, timeout=0):
    """
    Batch processes a list of Jupyter notebooks.

    Args:
        notebook_list (list): List of paths to Jupyter notebooks to process.
        output_dir (str): Directory to save the executed notebooks.
        timeout (int): Timeout (in seconds) for executing a single cell. Default is 600.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for notebook_path in notebook_list:
        try:
            # Read the notebook
            with open(notebook_path, 'r', encoding='utf-8') as nb_file:
                notebook = nbformat.read(nb_file, as_version=4)

            # Execute the notebook
            print(f"Processing notebook: {notebook_path}")
            ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
            ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})

            # Save the executed notebook to the output directory
            output_path = os.path.join(output_dir, os.path.basename(notebook_path.replace('.ipynb', '_out.ipynb')))
            with open(output_path, 'w', encoding='utf-8') as output_file:
                nbformat.write(notebook, output_file)
            print(f"Notebook processed and saved to: {output_path}")

        except Exception as e:
            print(f"Error processing notebook {notebook_path}: {e}")

if __name__ == "__main__":
    # Example usage
    notebooks_to_process = [
    "../notebooks/DenseNet131_gradual_unfreeze.ipynb",
    "../notebooks/DenseNet131-premult.ipynb",
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
    "../notebooks/Resnet18_upscale_unfreeze.ipynb",
    "../notebooks/Resnet50_gradual_unfreeze.ipynb",
    "../notebooks/Resnet50_premult.ipynb",
    "../notebooks/Resnet50_unfreeze.ipynb",
    "../notebooks/ViT_premult.ipynb",
    "../notebooks/ViT_unfreeze.ipynb",
    "../notebooks/ViT_unfreeze_gradual.ipynb"
]
    output_directory = "../notebooks"

    batch_process_notebooks(notebooks_to_process, output_directory)
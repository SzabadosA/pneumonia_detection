import os
import nbformat
from nbclient import NotebookClient

def execute_notebook_with_kernel_restart(notebook_path, output_path=None, timeout=600):
    """
    Executes a Jupyter notebook with kernel restart before running the cells.
    Args:
        notebook_path (str): Path to the input notebook file.
        output_path (str): Path to save the executed notebook. If None, overwrites the original file.
        timeout (int): Execution timeout in seconds.
    """
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook with kernel restart
    client = NotebookClient(nb, timeout=timeout, kernel_name='python3', restart_kernel=True)
    try:
        client.execute()
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}")

    # Save the executed notebook
    if output_path is None:
        output_path = notebook_path  # Overwrite original
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Executed notebook saved to {output_path}")

# List of notebooks to execute
notebooks = [
    "notebook1.ipynb",
    "notebook2.ipynb",
    "notebook3.ipynb"
]

# Execute each notebook with kernel restart
for notebook in notebooks:
    execute_notebook_with_kernel_restart(notebook)

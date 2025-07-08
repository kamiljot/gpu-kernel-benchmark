"""
new_kernel_op.py
----------------
Utility script for generating boilerplate files for a new kernel operation.

- Uses templates from the "templates" directory to create .h, .cuh, .cu, and launcher files.
- Usage: python3 new_kernel_op.py <kernel_name>

Author: Kamil J.
Date: 2025-07-07
"""

import os
import sys
from datetime import date

TEMPLATE_DIR = "templates"

def load_template(filename, name, current_date):
    """
    Loads a template file and replaces {{name}} and {{date}} with provided values.

    Args:
        filename (str): Template file name.
        name (str): Kernel name to substitute.
        current_date (str): Date string to substitute.

    Returns:
        str: The content of the template with substitutions.

    Raises:
        SystemExit: If the template file does not exist.
    """
    path = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: template file {filename} not found in {TEMPLATE_DIR}")
        sys.exit(1)
    with open(path, "r") as f:
        content = f.read()
    return content.replace("{{name}}", name).replace("{{date}}", current_date)

def main():
    """
    Main entry point: generates files for a new kernel operation from templates.
    """
    if len(sys.argv) < 2:
        print("Usage: python3 new_kernel_op.py <kernel_name>")
        sys.exit(1)

    kernel = sys.argv[1].lower()
    kernel_dir = os.path.join("src", "kernels", kernel)
    os.makedirs(kernel_dir, exist_ok=True)

    today = date.today().isoformat()  # 'YYYY-MM-DD'

    files = {
        f"{kernel}.h": load_template("kernel.h", kernel, today),
        f"{kernel}_kernels.cuh": load_template("kernel_kernels.cuh", kernel, today),
        f"{kernel}_kernels.cu": load_template("kernel_kernels.cu", kernel, today),
        f"{kernel}_launcher.cu": load_template("kernel_launcher.cu", kernel, today),
    }

    for fname, content in files.items():
        path = os.path.join(kernel_dir, fname)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)
            print(f"Created: {path}")
        else:
            print(f"Exists, skipping: {path}")

    # Here you can call your other update functions (kernel_dispatch, readme, cpu_baseline) if needed

if __name__ == "__main__":
    main()
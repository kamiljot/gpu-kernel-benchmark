import os
import sys

TEMPLATE_DIR = "templates"

def load_template(filename, name):
    path = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: template file {filename} not found in {TEMPLATE_DIR}")
        sys.exit(1)
    with open(path, "r") as f:
        content = f.read()
    return content.replace("{{name}}", name)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 new_kernel_op.py <kernel_name>")
        sys.exit(1)

    kernel = sys.argv[1].lower()
    kernel_dir = os.path.join("src", "kernels", kernel)
    os.makedirs(kernel_dir, exist_ok=True)

    files = {
        f"{kernel}.h": load_template("kernel.h", kernel),
        f"{kernel}_kernels.cuh": load_template("kernel_kernels.cuh", kernel),
        f"{kernel}_kernels.cu": load_template("kernel_kernels.cu", kernel),
        f"{kernel}_launcher.cu": load_template("kernel_launcher.cu", kernel),
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
import argparse
import numpy as np
import random

def check_alignment(csv_path, npy_path):
    # Load the CSV and NPY data
    csv_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    npy_data = np.load(npy_path)

    # Check the shape
    if csv_data.shape != npy_data.shape:
        return False, "Shape mismatch"

    # Check initial and final values
    if not np.array_equal(csv_data[:5], npy_data[:5]) or not np.array_equal(csv_data[-5:], npy_data[-5:]):
        return False, "Initial or final values mismatch"

    # Randomly check a few values
    for _ in range(10):
        i, j = random.randint(0, csv_data.shape[0] - 1), random.randint(0, csv_data.shape[1] - 1)
        if csv_data[i, j] != npy_data[i, j]:
            return False, f"Mismatch at position ({i}, {j})"
    
    # If all checks pass
    return True, "All values aligned"

def main():
    parser = argparse.ArgumentParser(description="Check alignment of CSV and NPY files")
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('npy_path', type=str, help='Path to the NPY file')
    args = parser.parse_args()

    is_aligned, message = check_alignment(args.csv_path, args.npy_path)
    if is_aligned:
        print(f"Values are aligned for {args.npy_path}")
    else:
        print(f"Values are NOT aligned for {args.npy_path}: {message}")

if __name__ == '__main__':
    main()

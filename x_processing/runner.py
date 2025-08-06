import os
from multiprocessing import get_context, set_start_method
from tqdm import tqdm
import glob
from x import init_worker, process_file

def find_files(input_dir):
    return list(glob.iglob(os.path.join(input_dir, "**", "*.csv.gz"), recursive=True))

def main():
    print("Runner started...")
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    files_to_process = find_files(input_dir)
    print(f"Found {len(files_to_process)} file(s).")

    if not files_to_process:
        print("No files found.")
        return

    num_workers = min(7, len(files_to_process))
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)))

    processed = sum(1 for r in results if r)
    print(f"{processed} file(s) processed.")

if __name__ == "__main__":
    set_start_method("spawn")
    main()

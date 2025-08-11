import os
import glob
import torch
from multiprocessing import get_context, set_start_method
from x import WorkerPool  
def find_files(input_dir):
    return list(glob.iglob(os.path.join(input_dir, "**", "*.csv.gz"), recursive=True))

def chunkify(lst, n):
    avg = len(lst) / float(n)
    chunks = []
    last = 0.0
    while last < len(lst):
        chunks.append(lst[int(last):int(last + avg)])
        last += avg
    return chunks

def worker_process_files(args):
    file_list, device_id = args
    worker = WorkerPool(device_id)
    results = worker.process_files(file_list)
    return results

def main():
    print("Runner started...")

    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "x-24-us-election"))
    files_to_process = find_files(input_dir)
    print(f"Found {len(files_to_process)} file(s).")

    if not files_to_process:
        print("No files to process.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available.")

    print(f"Using {num_gpus} CUDA device(s).")

    file_chunks = chunkify(files_to_process, num_gpus)
    args = [(chunk, i) for i, chunk in enumerate(file_chunks)]

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_gpus) as pool:
        results = pool.map(worker_process_files, args)

    total_processed = sum(len(r) for r in results if r)
    print(f"{total_processed} file(s) processed.")

if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()

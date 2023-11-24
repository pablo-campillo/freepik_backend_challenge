import time
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt

root_dir = Path(__file__).parent.parent
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def cpu_inference(number_of_images):
    print("cpu_inference")
    processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)
    cpu_model = AutoModelForCausalLM.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)

    file_name = "tests/data/cats.jpg"
    image = Image.open(Path(__file__).parent.parent / file_name)
    images = [image] * number_of_images

    t0 = time.perf_counter()
    pixel_values = processor(images=images, return_tensors="pt")
    pixel_values = pixel_values.pixel_values
    generated_ids = cpu_model.generate(pixel_values=pixel_values, max_length=50)
    _ = processor.batch_decode(generated_ids, skip_special_tokens=True)
    t1 = time.perf_counter()

    print("DONE")

    return t1 - t0

def performance_encoder_batch():
    """ Performance test to compare sequential and batch inference.

    Results are saved as a csv in docs/i2/batch_times.csv and docs/i5/batch_times.png.
    """
    batch_gpu_times = [0.5906854959903285, 2.2408461029990576, 4.653293856012169, 7.443443212017883]
    times = []
    for batch_gpu_time, number_of_images in zip(batch_gpu_times, range(1, 65, 16)):
        number_of_images = 1 if number_of_images == 1 else number_of_images - 1
        print(f"Number of images: {number_of_images}")
        with Pool(2) as p:
            cpu_times = p.map(cpu_inference, [1] * number_of_images)
        multi_cpu_t = sum(cpu_times)
        times.append((number_of_images, batch_gpu_time, multi_cpu_t))

    with (root_dir / 'docs/i5/batch_times.csv').open("w") as f:
        for number_of_images, batch_gpu_t, multi_cpu_t in times:
            f.writelines(f"{number_of_images},{batch_gpu_t},{multi_cpu_t}\n")

    fig, ax = plt.subplots()
    ax.plot(
        [number_of_images for number_of_images, _, _ in times], [batch_gpu_t for _, batch_gpu_t, _ in times],
        label = "Batch GPU"
    )
    ax.plot(
        [number_of_images for number_of_images, _, _ in times], [sequential_cpu_t for _, _, sequential_cpu_t in times],
        label = "Multi CPU (2 processes)"
    )
    ax.legend()
    ax.set_title("GPU Batch vs. Multi CPU (2 processes) performance increasing batch size")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time in seconds")
    plt.savefig(str(root_dir / 'docs/i5/batch_times.png'))


if __name__ == '__main__':
    performance_encoder_batch()

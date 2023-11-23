import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt


def performance_batch():
    """ Performance test to compare sequential and batch inference.

    Results are saved as a csv in docs/i2/batch_times.csv and docs/i2/batch_times.png.
    """
    root_dir = Path(__file__).parent.parent
    processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    file_name = "tests/data/cats.jpg"
    image = Image.open(Path(__file__).parent.parent / file_name)

    def inference(number_of_images):
        images = [image] * number_of_images
        t0 = time.perf_counter()
        pixel_values = processor(images=images, return_tensors="pt").to(device).pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        _ = processor.batch_decode(generated_ids, skip_special_tokens=True)
        t1 = time.perf_counter()

        return t1 - t0

    times = []
    for number_of_images in range(1, 65, 16):
        number_of_images = 1 if number_of_images == 1 else number_of_images - 1
        batch_t = inference(number_of_images)
        sequential_t = sum(inference(1) for _ in range(number_of_images))
        times.append((number_of_images, sequential_t, batch_t))

    with (root_dir / 'docs/i2/batch_times.csv').open("w") as f:
        for number_of_images, sequential_t, batch_t in times:
            f.writelines(f"{number_of_images},{sequential_t},{batch_t}\n")

    fig, ax = plt.subplots()
    ax.plot([number_of_images for number_of_images, _, _ in times], [sequential_t for _, sequential_t, _ in times], label = "Sequential")
    ax.plot([number_of_images for number_of_images, _, _ in times], [batch_t for _, _, batch_t in times], label = "Batch")
    ax.legend()
    ax.set_title("GPU performance increasing batch size")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time in seconds")
    plt.savefig(str(root_dir / 'docs/i2/batch_times.png'))


if __name__ == '__main__':
    performance_batch()

import asyncio
from pathlib import Path

import torch
from decouple import config
from transformers import AutoModelForCausalLM, AutoProcessor

MAX_BATCH_SIZE = config('MAX_BATCH_SIZE', 64, cast=int)


async def server_loop(q):
    """Thread service that batches requests for caption generation from images.

    :param q: queue where inference requests are received
    :return:
    """
    root_dir = Path(__file__).parent.parent.parent.parent
    processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True).to(device)

    while True:
        (image, response_q) = await q.get()
        images = [image]
        queues = [response_q]
        while True:
            try:
                (image, response_q) = await asyncio.wait_for(q.get(), timeout=0.001)  # 1ms
            except asyncio.exceptions.TimeoutError:
                break
            images.append(image)
            queues.append(response_q)
            if len(images) > MAX_BATCH_SIZE:
                break

        pixel_values = processor(images=images, return_tensors="pt").to(device).pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for response_q, generated_caption in zip(queues, generated_captions):
            await response_q.put(generated_caption)

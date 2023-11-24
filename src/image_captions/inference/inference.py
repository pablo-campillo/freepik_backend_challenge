import asyncio
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


MAX_BATCH_SIZE = 64


async def server_loop(q):
    root_dir = Path(__file__).parent.parent.parent.parent
    processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True).to(device)

    while True:
        (token, response_q) = await q.get()
        if token is None:
            break
        pixel_values_serialized = np.frombuffer(token, dtype=np.float32)
        pixel_values = pixel_values_serialized.reshape((1, 3, 224, 224))
        pixel_values = [pixel_values[0]]
        queues = [response_q]
        while True:
            try:
                (token, response_q) = await asyncio.wait_for(q.get(), timeout=0.001)  # 1ms
            except asyncio.exceptions.TimeoutError:
                break
            pixel_values_serialized = np.frombuffer(token, dtype=np.float32)
            pixel_values.append(pixel_values_serialized.reshape((1, 3, 224, 224))[0])
            queues.append(response_q)
            if len(pixel_values) > MAX_BATCH_SIZE:
                break

        pixel_values = np.array(pixel_values)
        pixel_values = torch.tensor(pixel_values).to(device)
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for response_q, generated_caption in zip(queues, generated_captions):
            await response_q.put(generated_caption)
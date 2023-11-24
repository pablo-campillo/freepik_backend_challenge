import time
from pathlib import Path

import pytest
import numpy as np
import torch

from PIL import Image


@pytest.mark.parametrize(
    "file_name,words_expected",
    [
        ("tests/data/cats.jpg", ["two", "cats", "sleeping", "couch"]),
        ("tests/data/dog.jpg", ["dog", "running", "grass"]),
    ]
)
def test_model_examples(processor, model, file_name, words_expected):
    image = Image.open(Path(__file__).parent.parent / file_name)

    processor1_t0 = time.perf_counter()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    processor1_t1 = time.perf_counter()

    model_t0 = time.perf_counter()
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    model_t1 = time.perf_counter()
    processor2_t0 = time.perf_counter()
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processor2_t1 = time.perf_counter()

    print(f"t1: {processor1_t1 - processor1_t0}")
    print(f"t2: {model_t1 - model_t0}")
    print(f"t3: {processor2_t1 - processor2_t0}")

    words = generated_caption.split(' ')
    for we in words_expected:
        assert we in words


@pytest.mark.parametrize(
    "file_name,words_expected",
    [
        ("tests/data/cats.jpg", ["two", "cats", "sleeping", "couch"]),
        ("tests/data/dog.jpg", ["dog", "running", "grass"]),
    ]
)
def test_model_numpy_serialization(processor, model, file_name, words_expected):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    image = Image.open(Path(__file__).parent.parent / file_name)

    processor1_t0 = time.perf_counter()
    pixel_values = processor(images=image, return_tensors="np").pixel_values
    serialized = pixel_values.tobytes()
    pixel_values_serialized = np.frombuffer(serialized, dtype=np.float32)
    pixel_values = torch.tensor(pixel_values_serialized.reshape((1, 3, 224, 224))).to(device)
    processor1_t1 = time.perf_counter()

    model_t0 = time.perf_counter()
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    model_t1 = time.perf_counter()
    processor2_t0 = time.perf_counter()
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processor2_t1 = time.perf_counter()

    print(f"t1: {processor1_t1 - processor1_t0}")
    print(f"t2: {model_t1 - model_t0}")
    print(f"t3: {processor2_t1 - processor2_t0}")

    words = generated_caption.split(' ')
    for we in words_expected:
        assert we in words
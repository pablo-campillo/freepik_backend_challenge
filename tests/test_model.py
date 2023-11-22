from pathlib import Path

import pytest

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

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    words = generated_caption.split(' ')
    for we in words_expected:
        assert we in words

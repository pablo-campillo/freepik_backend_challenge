from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

root_dir = Path(__file__).parent.parent
processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)


def encode_image(file_path):
    image = Image.open(file_path)

    pixel_values = processor(images=image, return_tensors="np").pixel_values
    serialized = pixel_values.tobytes()

    with open('../tests/data/encoded_cats.bin', "wb") as f:
        f.write(serialized)


if __name__ == '__main__':
    encode_image('../tests/data/cats.jpg')


import asyncio
from pathlib import Path

import pytest
import tornado
from tornado.ioloop import IOLoop
from PIL import Image

from image_captions.inference.handler import InferenceReceiverHandler
from image_captions.inference.inference import server_loop


@pytest.fixture(scope='module')
def model_queue():
    return asyncio.Queue()


@pytest.fixture
def app(model_queue):
    handlers = [
        (r"/", InferenceReceiverHandler, dict(model_queue=model_queue)),
    ]
    settings = {
        'debug': False
    }
    return tornado.web.Application(handlers, **settings)


async def test_inference_api(http_server_client, processor, model_queue):
    IOLoop.current().spawn_callback(server_loop, model_queue)
    file_name = "tests/data/cats.jpg"
    image = Image.open(Path(__file__).parent.parent.parent / file_name)

    pixel_values = processor(images=image, return_tensors="np").pixel_values
    serialized = pixel_values.tobytes()

    response = await http_server_client.fetch("/", method='POST', body=serialized)
    assert response.code == 200
    words = response.body.split(' ')
    for we in ["two", "cats", "sleeping", "couch"]:
        assert we in words

    await model_queue.put((None, None))
    await asyncio.sleep(1)

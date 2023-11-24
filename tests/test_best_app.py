import asyncio
from pathlib import Path

import pytest
import tornado
from PIL import Image
from tornado.ioloop import IOLoop

from image_captions.best.handler import ImageReceiverHandler
from image_captions.best.inference import server_loop


@pytest.fixture(scope='module')
def model_queue():
    return asyncio.Queue()


@pytest.fixture
def app(model_queue):
    handlers = [
        (r"/", ImageReceiverHandler, dict(model_queue=model_queue)),
    ]
    settings = {
        'debug': False
    }
    return tornado.web.Application(handlers, **settings)


async def test_best_app(http_server_client, processor, model_queue):
    IOLoop.current().spawn_callback(server_loop, model_queue)
    file_name = "tests/data/cats.jpg"
    image = Image.open(Path(__file__).parent.parent / file_name)

    response = await http_server_client.fetch("/", method='POST', body=image.tobytes())
    assert response.code == 200
    words = response.body.split(' ')
    for we in ["two", "cats", "sleeping", "couch"]:
        assert we in words

    await model_queue.put((None, None))
    await asyncio.sleep(1)

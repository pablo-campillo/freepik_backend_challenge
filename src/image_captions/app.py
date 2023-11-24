
from pathlib import Path

import tornado.web
from tornado import httpclient
from tornado.httpserver import HTTPServer
from tornado.netutil import bind_sockets
import io

import asyncio

import sys

from PIL import Image
from transformers import AutoProcessor

PORT = 8888
ADDR = '0.0.0.0'
MAX_BATCH_SIZE = 64

sockets = bind_sockets(PORT, ADDR)
tornado.process.fork_processes(tornado.process.cpu_count() - 1)

root_dir = Path(__file__).parent.parent.parent
processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)


def encode_image(file_path):
    image = Image.open(file_path)


async def get_captions(image_bytes):
    http_client = httpclient.AsyncHTTPClient()
    pixel_values = processor(images=image_bytes, return_tensors="np").pixel_values
    serialized = pixel_values.tobytes()
    try:
        response = await http_client.fetch(
            "http://0.0.0.0:8889/",
            method="POST",
            body=serialized,
        )
        return response.body.decode('utf-8')
    except httpclient.HTTPError as e:
        # HTTPError is raised for non-200 responses; the response
        # can be found in e.response.
        print("Error: " + str(e))
    except Exception as e:
        # Other errors are possible, such as IOError.
        print("Error: " + str(e))
    http_client.close()


class ImageReceiverHandler(tornado.web.RequestHandler):
    async def post(self):
        generated_caption = await get_captions(Image.open(io.BytesIO(self.request.body)))
        self.write(generated_caption)


async def main():
    handlers = [
        (r"/", ImageReceiverHandler),
    ]
    debug = bool(sys.flags.debug)
    settings = {
        'debug': debug
    }
    application = tornado.web.Application(handlers, **settings)
    server = HTTPServer(application)
    server.add_sockets(sockets)

    print(f"server listening at server:port debug={debug}")
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()


if __name__ == "__main__":
    asyncio.run(main())

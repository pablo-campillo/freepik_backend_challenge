from pathlib import Path

import torch
import tornado.web
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.netutil import bind_sockets
import io

import asyncio

import sys

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

PORT = 8888
ADDR = '0.0.0.0'

sockets = bind_sockets(PORT, ADDR)
# tornado.process.fork_processes(tornado.process.cpu_count() - 1)
tornado.process.fork_processes(1)

root_dir = Path(__file__).parent.parent.parent
processor = AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True).to(device)


# @tornado.web.stream_request_body
class ImageReceiverHandler(tornado.web.RequestHandler):
    async def post(self):
        image = Image.open(io.BytesIO(self.request.body))

        pixel_values = processor(images=image, return_tensors="pt").to(device).pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # IOLoop.current().spawn_callback(send_rows, rows)

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

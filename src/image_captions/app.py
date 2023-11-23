
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
MAX_BATCH_SIZE = 64

sockets = bind_sockets(PORT, ADDR)
# tornado.process.fork_processes(tornado.process.cpu_count() - 1)
tornado.process.fork_processes(1)


# @tornado.web.stream_request_body
class ImageReceiverHandler(tornado.web.RequestHandler):
    def initialize(self, model_queue):
        self.model_queue = model_queue

    async def post(self):
        response_q = asyncio.Queue()
        await self.model_queue.put((Image.open(io.BytesIO(self.request.body)), response_q))
        generated_caption = await response_q.get()

        self.write(generated_caption)


async def server_loop(q):
    root_dir = Path(__file__).parent.parent.parent
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


async def main():
    model_queue = asyncio.Queue()
    handlers = [
        (r"/", ImageReceiverHandler, dict(model_queue=model_queue)),
    ]
    debug = bool(sys.flags.debug)
    settings = {
        'debug': debug
    }
    application = tornado.web.Application(handlers, **settings)
    server = HTTPServer(application)
    server.add_sockets(sockets)

    IOLoop.current().spawn_callback(server_loop, model_queue)

    print(f"server listening at server:port debug={debug}")
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()


if __name__ == "__main__":
    asyncio.run(main())

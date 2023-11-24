import asyncio
import io

import tornado
from PIL import Image


class ImageReceiverHandler(tornado.web.RequestHandler):
    """ Handler for POST requests

    It processes a post request with an image and returns its caption.
    It asks for the caption through a queue to the inference function.
    """
    def initialize(self, model_queue):
        self.model_queue = model_queue

    async def post(self):
        response_q = asyncio.Queue()
        await self.model_queue.put((Image.open(io.BytesIO(self.request.body)), response_q))
        generated_caption = await response_q.get()

        self.write(generated_caption)

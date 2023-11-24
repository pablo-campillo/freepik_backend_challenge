import asyncio

import tornado


class InferenceReceiverHandler(tornado.web.RequestHandler):
    def initialize(self, model_queue):
        self.model_queue = model_queue

    async def post(self):
        response_q = asyncio.Queue()
        await self.model_queue.put((self.request.body, response_q))
        generated_caption = await response_q.get()

        self.write(generated_caption)

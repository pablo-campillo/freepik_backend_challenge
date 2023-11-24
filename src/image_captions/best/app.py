
import asyncio
import sys

import tornado.web
from decouple import config
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.netutil import bind_sockets

from image_captions.best.handler import ImageReceiverHandler
from image_captions.best.inference import server_loop

PORT = config('PORT', 8888, cast=int)
ADDR = config('ADDR', '0.0.0.0')

sockets = bind_sockets(PORT, ADDR)
tornado.process.fork_processes(1)


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

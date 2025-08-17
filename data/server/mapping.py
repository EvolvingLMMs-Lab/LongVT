from .base_server import BaseServer

SERVER_MAPPING = {}


# A decorator to register a server class
def register_server(server_class, name: str):
    SERVER_MAPPING[name] = server_class
    return server_class


def get_server(name: str) -> BaseServer:
    return SERVER_MAPPING[name]

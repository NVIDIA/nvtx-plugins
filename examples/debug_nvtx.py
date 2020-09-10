import nvtx
import time

DEFAULT_DOMAIN = "test-domain"


class BaseCallback(object):

    def open_marker(self, message):
        print("Pushing: `{}` ...".format(message))
        nvtx.push_range(message=message, color="blue", domain=DEFAULT_DOMAIN)

    def close_marker(self):
        print("Popping range ...")
        nvtx.pop_range(domain=DEFAULT_DOMAIN)


if __name__ == "__main__":

    foo = BaseCallback()
    foo.open_marker("blah")
    time.sleep(2)
    foo.close_marker()

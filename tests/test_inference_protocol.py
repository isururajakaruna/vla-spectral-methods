import socket
import unittest

import numpy as np

from smooth_operator.inference_protocol import (
    decode_request,
    encode_request,
    receive_message,
    send_message,
)


class InferenceProtocolTests(unittest.TestCase):
    def test_request_round_trip(self):
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        state = np.asarray((0.1, -0.2, -0.3, 1.0), dtype=np.float32)
        left, right = socket.socketpair()
        try:
            send_message(left, encode_request(image, state, "Move the apple"))
            decoded_image, decoded_state, prompt = decode_request(receive_message(right))
        finally:
            left.close()
            right.close()
        np.testing.assert_array_equal(decoded_image, image)
        np.testing.assert_allclose(decoded_state, state)
        self.assertEqual(prompt, "Move the apple")

    def test_action_response_round_trip(self):
        actions = np.zeros((10, 4), dtype=float).tolist()
        left, right = socket.socketpair()
        try:
            send_message(left, {"actions": actions})
            self.assertEqual(receive_message(right)["actions"], actions)
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()

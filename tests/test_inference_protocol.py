import socket
import unittest

import numpy as np

from smooth_operator.inference_protocol import (
    decode_request,
    encode_multi_request,
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

    def test_multi_camera_request_round_trip(self):
        images = {
            f"observation.images.camera{index}": np.full(
                (8, 10, 3), index, dtype=np.uint8
            )
            for index in range(1, 4)
        }
        state = np.arange(6, dtype=np.float32)
        decoded_images, decoded_state, prompt = decode_request(
            encode_multi_request(images, state, "Move the apple")
        )
        self.assertEqual(set(decoded_images), set(images))
        for key in images:
            np.testing.assert_array_equal(decoded_images[key], images[key])
        np.testing.assert_array_equal(decoded_state, state)
        self.assertEqual(prompt, "Move the apple")


if __name__ == "__main__":
    unittest.main()

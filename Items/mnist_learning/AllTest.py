import unittest
import imageio
import requests


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_Image_first(self):
        img_arr = imageio.imread_v2('mnist_train/0/mnist_train_1.png')
        print(img_arr.shape)



if __name__ == '__main__':
    unittest.main()

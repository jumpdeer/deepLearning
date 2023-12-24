import os
import unittest
import imageio

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_Image(self):
        img_arr = imageio.imread_v2('archive/dataset-master/dataset-master/JPEGImages/BloodImage_00000.jpg')
        print(img_arr.shape)


    def test_os_walk(self):
        image_path = 'archive/dataset-master/dataset-master/JPEGImages/'
        annotation_path = 'archive/dataset-master/dataset-master/Annotations/'
        LabelImage_path = 'archive/dataset-master/dataset-master/Label/'
        image_files = []
        for root, dirs, files in os.walk(image_path):
            image_files = files
        print(image_files)


if __name__ == '__main__':
    unittest.main()

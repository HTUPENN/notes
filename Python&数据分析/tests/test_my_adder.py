import unittest
from files.myadder import my_adder

class TestMyadder(unittest.TestCase): # 测试方法必须以 test_ 开头
    def test_positive_with_positive(self): # must start with test_
        self.assertEqual(my_adder(3,5), 7)

    def test_negative_with_positive(self):
        self.assertEqual(my_adder(-3,5), 2)
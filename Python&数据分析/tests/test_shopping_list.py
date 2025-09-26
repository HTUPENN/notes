import unittest
from files.shopping_list import ShoppingList


class MyTestCase(unittest.TestCase):
    def setUp(self):  # 每个函数运行前都会被实例
        self.shopping_list = ShoppingList({"item1": 8, "item2": 30, "item3": 15})

    def test_get_item_count(self):
        self.assertEqual(self.shopping_list.get_items_count(), 3)  # add assertion here

    def test_get_total_price(self):
        self.assertEqual(self.shopping_list.get_total_price(), 50)  # 53


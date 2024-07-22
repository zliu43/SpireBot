import unittest
from tensorgenerator import TensorGenerator

class MyTestCase(unittest.TestCase):

    tensorgenerator = TensorGenerator()
    hand = ['hand_card1','hand_card2','hand_card3','hand_card4','hand_card5']
    discard = ['discard_card1','discard_card1','discard_card3','discard_card4','discard_card5']
    enemies = ['enemy_1','enemy2','enemy3']

    tensor2move_test_cases = {
        0:  ['end_turn'],
        25: ['play hand_card5 '],
        21: ['play hand_card4 enemy2'],
        67: ['hand_card1'],
        93: ['hand_card3'],
        101:['discard_card1']
    }

    def test_tensor2move(self):
        for test in self.tensor2move_test_cases:
            result = self.tensorgenerator.tensor2move(test, self.hand, self.enemies, self.discard)
            self.assertEqual(self.tensor2move_test_cases[test], result)

if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from engine.solver._solver import BaseSolver


class TuningStateTests(unittest.TestCase):
    def test_wholebody49_to69_score_head_copies_only_existing_class_rows(self):
        cur_tensor = torch.full((70, 4), -1.0)
        pretrain_tensor = torch.arange(50 * 4, dtype=torch.float32).reshape(50, 4)

        adjusted, info = BaseSolver._map_wholebody49_to69_score_head(cur_tensor, pretrain_tensor)

        self.assertIsNotNone(adjusted)
        self.assertTrue(torch.equal(adjusted[:49], pretrain_tensor[:49]))
        self.assertTrue(torch.equal(adjusted[49:], cur_tensor[49:]))
        self.assertEqual(info['copied_semantic_rows'], 49)
        self.assertEqual(info['kept_initialized_rows'], 21)

    def test_wholebody49_to69_score_bias_copies_only_existing_class_rows(self):
        cur_tensor = torch.full((70,), -1.0)
        pretrain_tensor = torch.arange(50, dtype=torch.float32)

        adjusted, info = BaseSolver._map_wholebody49_to69_score_head(cur_tensor, pretrain_tensor)

        self.assertIsNotNone(adjusted)
        self.assertTrue(torch.equal(adjusted[:49], pretrain_tensor[:49]))
        self.assertTrue(torch.equal(adjusted[49:], cur_tensor[49:]))
        self.assertEqual(info['copied_semantic_rows'], 49)
        self.assertEqual(info['kept_initialized_rows'], 21)

    def test_wholebody49_to69_denoising_embed_copies_existing_classes_and_padding(self):
        cur_tensor = torch.full((71, 4), -1.0)
        pretrain_tensor = torch.arange(51 * 4, dtype=torch.float32).reshape(51, 4)

        adjusted, info = BaseSolver._map_wholebody49_to69_denoising_embed(cur_tensor, pretrain_tensor)

        self.assertIsNotNone(adjusted)
        self.assertTrue(torch.equal(adjusted[:49], pretrain_tensor[:49]))
        self.assertTrue(torch.equal(adjusted[49:70], cur_tensor[49:70]))
        self.assertTrue(torch.equal(adjusted[70], pretrain_tensor[50]))
        self.assertEqual(info['copied_semantic_rows'], 49)
        self.assertEqual(info['kept_initialized_rows'], 21)
        self.assertEqual(info['copied_padding_row'], (50, 70))

    def test_adjust_head_parameters_removes_unmapped_denoising_shape(self):
        solver = BaseSolver.__new__(BaseSolver)
        solver.obj365_ids = []
        cur_state = {
            'decoder.denoising_class_embed.weight': torch.zeros((10, 4)),
        }
        pretrain_state = {
            'decoder.denoising_class_embed.weight': torch.ones((8, 4)),
        }

        adjusted_state, adjust_infos = solver._adjust_head_parameters(cur_state, pretrain_state)

        self.assertNotIn('decoder.denoising_class_embed.weight', adjusted_state)
        self.assertEqual(adjust_infos, [])
        self.assertIn('decoder.denoising_class_embed.weight', pretrain_state)


if __name__ == '__main__':
    unittest.main()

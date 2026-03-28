import random
import unittest

import numpy as np
import torch
from PIL import Image

from engine.data.dataloader import BatchImageCollateFunction
from engine.data.transforms.container import Compose
from engine.data.transforms.mosaic import Mosaic
from engine.misc import dist_utils
from engine.optim.ema import ModelEMA
from engine.optim.lr_scheduler import FlatCosineLRScheduler


class ResumeStateTests(unittest.TestCase):
    def test_model_ema_ignores_extra_state_and_copies_non_floating_tensors(self):
        class ToyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
                self.register_buffer('counter', torch.tensor([1], dtype=torch.int64))
                self.tag = 'initial'

            def get_extra_state(self):
                return {'tag': self.tag}

            def set_extra_state(self, state):
                if state:
                    self.tag = state.get('tag', self.tag)

        model = ToyModule()
        ema = ModelEMA(model, decay=0.5, warmups=0)

        model.weight.data.fill_(3.0)
        model.counter.fill_(7)
        model.tag = 'updated'

        ema.update(model)

        ema_state = ema.module.state_dict()
        self.assertTrue(torch.allclose(ema_state['weight'], torch.tensor([2.0])))
        self.assertTrue(torch.equal(ema_state['counter'], torch.tensor([7], dtype=torch.int64)))
        self.assertEqual(ema_state['_extra_state'], {'tag': 'initial'})

    def test_flat_cosine_scheduler_state_roundtrip(self):
        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([param], lr=0.1)
        optimizer.param_groups[0]['initial_lr'] = 0.1
        scheduler = FlatCosineLRScheduler(
            optimizer,
            lr_gamma=0.5,
            iter_per_epoch=4,
            total_epochs=5,
            warmup_iter=2,
            flat_epochs=2,
            no_aug_epochs=1,
        )
        state = scheduler.state_dict()

        new_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
        new_optimizer.param_groups[0]['initial_lr'] = 0.1
        restored = FlatCosineLRScheduler(
            new_optimizer,
            lr_gamma=0.1,
            iter_per_epoch=1,
            total_epochs=1,
            warmup_iter=0,
            flat_epochs=0,
            no_aug_epochs=0,
        )
        restored.load_state_dict(state)

        for step in [0, 1, 2, 5, 9, 12]:
            optimizer = scheduler.step(step, optimizer)
            new_optimizer = restored.step(step, new_optimizer)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], new_optimizer.param_groups[0]['lr'])

    def test_collate_state_roundtrip(self):
        collate = BatchImageCollateFunction(mixup_prob=0.5, copyblend_prob=0.5)
        collate.set_epoch(7)
        collate.print_info_flag = False
        collate.print_copyblend_flag = False

        restored = BatchImageCollateFunction(mixup_prob=0.5, copyblend_prob=0.5)
        restored.load_state_dict(collate.state_dict())

        self.assertEqual(restored.epoch, 7)
        self.assertFalse(restored.print_info_flag)
        self.assertFalse(restored.print_copyblend_flag)

    def test_compose_and_mosaic_state_roundtrip(self):
        compose = Compose(ops=None)
        compose.global_samples = 11
        restored = Compose(ops=None)
        restored.load_state_dict(compose.state_dict())
        self.assertEqual(restored.global_samples, 11)

        mosaic = Mosaic(output_size=32, use_cache=True, max_cached_images=4)
        image = Image.new('RGB', (8, 8), color=0)
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.int64),
        }
        mosaic.mosaic_cache = [{'img': image, 'labels': target}]
        state = mosaic.state_dict()

        restored_mosaic = Mosaic(output_size=32, use_cache=True, max_cached_images=4)
        restored_mosaic.load_state_dict(state)

        self.assertEqual(len(restored_mosaic.mosaic_cache), 1)
        self.assertEqual(restored_mosaic.mosaic_cache[0]['img'].size, image.size)
        self.assertEqual(restored_mosaic.mosaic_cache[0]['labels']['boxes'].shape, target['boxes'].shape)

    def test_rng_state_roundtrip(self):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)

        state = dist_utils.capture_rng_state()
        expected = (
            random.random(),
            np.random.rand(),
            torch.rand(2),
        )
        expected_cuda = torch.rand(2, device='cuda') if torch.cuda.is_available() else None

        dist_utils.restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(2), expected[2]))
        if torch.cuda.is_available():
            self.assertTrue(torch.equal(torch.rand(2, device='cuda'), expected_cuda))


if __name__ == '__main__':
    unittest.main()

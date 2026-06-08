import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from engine.core import YAMLConfig
from engine.data.dataloader import BatchImageCollateFunction
from engine.data.transforms.container import Compose
from engine.data.transforms.mosaic import Mosaic
from engine.misc import dist_utils
from engine.optim.ema import ModelEMA
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from engine.solver.det_solver import DetSolver


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

    def test_stage2_checkpoint_source_config_defaults_and_override(self):
        repo_root = Path(__file__).resolve().parents[1]

        default_cfg = YAMLConfig(str(repo_root / 'configs/base/deimv2.yml'))
        self.assertEqual(default_cfg.stage2_checkpoint_source, 'best')

        wholebody69_cfg = YAMLConfig(
            str(repo_root / 'configs/deimv2/deimv2_dinov3_x_wholebody69_ins_s08_maskhead256x3_center.yml')
        )
        self.assertEqual(wholebody69_cfg.stage2_checkpoint_source, 'last')

    def test_stage2_checkpoint_source_validation_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            solver = DetSolver.__new__(DetSolver)
            solver.output_dir = output_dir

            solver.cfg = SimpleNamespace(stage2_checkpoint_source='best')
            self.assertEqual(solver._get_stage2_checkpoint_path(), output_dir / 'best_stg1.pth')

            solver.cfg = SimpleNamespace(stage2_checkpoint_source='last')
            with self.assertRaises(FileNotFoundError):
                solver._get_stage2_checkpoint_path()

            fallback = output_dir / 'last.pth'
            fallback.touch()
            self.assertEqual(solver._get_stage2_checkpoint_path(), fallback)

            last_stg1 = output_dir / 'last_stg1.pth'
            last_stg1.touch()
            self.assertEqual(solver._get_stage2_checkpoint_path(), last_stg1)

            solver.cfg = SimpleNamespace(stage2_checkpoint_source='invalid')
            with self.assertRaises(ValueError):
                solver._get_stage2_checkpoint_source()

    def test_last_stage2_reload_preserves_current_epoch_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / 'last_stg1.pth').touch()

            solver = DetSolver.__new__(DetSolver)
            solver.cfg = SimpleNamespace(stage2_checkpoint_source='last')
            solver.output_dir = output_dir
            solver.last_epoch = 49

            loaded_paths = []
            set_epochs = []
            solver.load_resume_state = lambda path: loaded_paths.append(Path(path).name)
            solver.train_dataloader = SimpleNamespace(
                set_epoch=lambda epoch: set_epochs.append(epoch),
                sampler=SimpleNamespace(set_epoch=lambda epoch: None),
            )

            solver._load_stage2_checkpoint(50, preserve_last_epoch=True)

            self.assertEqual(loaded_paths, ['last_stg1.pth'])
            self.assertEqual(set_epochs, [50])
            self.assertEqual(solver.last_epoch, 50)


if __name__ == '__main__':
    unittest.main()

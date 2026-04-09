from types import SimpleNamespace

import pytest
import torch

from faster_coco_eval import COCO

from engine.data.dataset.coco_eval import CocoEvaluator
from engine.misc import dist_utils
from engine.solver.det_engine import evaluate
from engine.solver.det_solver import DetSolver


def build_coco_gt():
    coco = COCO()
    coco.dataset = {
        'images': [{'id': 1, 'width': 32, 'height': 32}],
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                'category_id': 21,
                'bbox': [0.0, 0.0, 10.0, 10.0],
                'area': 100.0,
                'iscrowd': 0,
            }
        ],
        'categories': [
            {'id': 21, 'name': 'collarbone'},
            {'id': 22, 'name': 'shoulder'},
        ],
    }
    coco.createIndex()
    return coco


def make_prediction(label, score, box):
    return {
        'labels': torch.tensor([label], dtype=torch.int64),
        'scores': torch.tensor([score], dtype=torch.float32),
        'boxes': torch.tensor([box], dtype=torch.float32),
    }


def test_center_evaluator_reports_perfect_match():
    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=[],
        center_eval_class_ids=[21],
        center_eval_thresholds=[0.25, 0.5, 1.0],
        center_eval_primary_threshold=0.5,
    )

    evaluator.update({1: make_prediction(21, 0.9, [4.0, 4.0, 6.0, 6.0])})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    summary = evaluator.center_eval['summary']
    assert summary['precision@0.5'] == pytest.approx(1.0)
    assert summary['recall@0.5'] == pytest.approx(1.0)
    assert summary['f1@0.5'] == pytest.approx(1.0)
    assert summary['median_distance'] == pytest.approx(0.0)


def test_center_evaluator_counts_miss_outside_threshold():
    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=[],
        center_eval_class_ids=[21],
        center_eval_thresholds=[0.5],
        center_eval_primary_threshold=0.5,
    )

    evaluator.update({1: make_prediction(21, 0.9, [20.0, 20.0, 22.0, 22.0])})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    summary = evaluator.center_eval['summary']
    assert summary['precision@0.5'] == pytest.approx(0.0)
    assert summary['recall@0.5'] == pytest.approx(0.0)
    assert summary['f1@0.5'] == pytest.approx(0.0)
    assert summary['matched_count'] == 0


def test_center_evaluator_marks_duplicate_predictions_as_false_positive():
    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=[],
        center_eval_class_ids=[21],
        center_eval_thresholds=[0.5],
        center_eval_primary_threshold=0.5,
    )

    evaluator.update({
        1: {
            'labels': torch.tensor([21, 21], dtype=torch.int64),
            'scores': torch.tensor([0.9, 0.8], dtype=torch.float32),
            'boxes': torch.tensor([[4.0, 4.0, 6.0, 6.0], [4.5, 4.5, 6.5, 6.5]], dtype=torch.float32),
        }
    })
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    summary = evaluator.center_eval['summary']
    assert summary['precision@0.5'] == pytest.approx(0.5)
    assert summary['recall@0.5'] == pytest.approx(1.0)
    assert summary['f1@0.5'] == pytest.approx(2.0 / 3.0)


def test_center_evaluator_requires_class_match():
    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=[],
        center_eval_class_ids=[21, 22],
        center_eval_thresholds=[0.5],
        center_eval_primary_threshold=0.5,
    )

    evaluator.update({1: make_prediction(22, 0.9, [4.0, 4.0, 6.0, 6.0])})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    summary = evaluator.center_eval['summary']
    assert summary['precision@0.5'] == pytest.approx(0.0)
    assert summary['recall@0.5'] == pytest.approx(0.0)
    assert summary['f1@0.5'] == pytest.approx(0.0)


def test_center_evaluator_is_disabled_with_empty_class_list():
    evaluator = CocoEvaluator(build_coco_gt(), iou_types=[], center_eval_class_ids=[])
    evaluator.update({1: make_prediction(21, 0.9, [4.0, 4.0, 6.0, 6.0])})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    assert evaluator.center_eval['summary'] == {}


def test_center_evaluator_merges_records_from_all_gather(monkeypatch):
    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=[],
        center_eval_class_ids=[21],
        center_eval_thresholds=[0.5],
        center_eval_primary_threshold=0.5,
    )
    evaluator.center_eval_records = {
        1: {'predictions': [], 'targets': []},
    }

    monkeypatch.setattr(
        dist_utils,
        'all_gather',
        lambda data: [
            data,
            {
                2: {
                    'predictions': [{'category_id': 21, 'score': 0.5, 'center': [1.0, 1.0]}],
                    'targets': [{'category_id': 21, 'center': [1.0, 1.0], 'bbox': [0.0, 0.0, 2.0, 2.0]}],
                }
            },
        ],
    )

    evaluator.synchronize_between_processes()
    assert sorted(evaluator.center_eval_records.keys()) == [1, 2]


def test_det_engine_evaluate_adds_center_stats():
    class DummyModel(torch.nn.Module):
        def forward(self, samples):
            return {'pred_logits': torch.zeros((1, 1, 41)), 'pred_boxes': torch.zeros((1, 1, 4))}

    class DummyPostProcessor:
        def __call__(self, outputs, orig_target_sizes):
            return [{
                'labels': torch.tensor([21], dtype=torch.int64),
                'scores': torch.tensor([0.9], dtype=torch.float32),
                'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            }]

    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=['bbox'],
        center_eval_class_ids=[21],
        center_eval_thresholds=[0.5],
        center_eval_primary_threshold=0.5,
    )
    data_loader = [(
        torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        [{
            'image_id': torch.tensor([1], dtype=torch.int64),
            'orig_size': torch.tensor([32, 32], dtype=torch.int64),
        }],
    )]

    stats, updated_evaluator = evaluate(
        DummyModel(),
        torch.nn.Identity(),
        DummyPostProcessor(),
        data_loader,
        evaluator,
        torch.device('cpu'),
    )

    assert 'coco_eval_bbox' in stats
    assert stats['center_eval_f1_at_0_5'] == pytest.approx(1.0)
    assert updated_evaluator.center_eval['summary']['f1@0.5'] == pytest.approx(1.0)


def test_det_solver_center_report_does_not_break_existing_validation_output(capsys):
    evaluator = CocoEvaluator(
        build_coco_gt(),
        iou_types=[],
        center_eval_class_ids=[21],
        center_eval_thresholds=[0.5],
        center_eval_primary_threshold=0.5,
    )
    evaluator.update({1: make_prediction(21, 0.9, [4.0, 4.0, 6.0, 6.0])})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    solver = object.__new__(DetSolver)
    label2category = {idx: idx for idx in range(23)}
    solver.val_dataloader = SimpleNamespace(dataset=SimpleNamespace(
        label2category=label2category,
        category2name={idx: str(idx) for idx in range(23)} | {21: 'collarbone', 22: 'shoulder'},
    ))

    solver._report_validation(evaluator, epoch=3)
    captured = capsys.readouterr()
    assert 'Center-point metrics:' in captured.out
    assert 'Per-class center-point metrics:' in captured.out

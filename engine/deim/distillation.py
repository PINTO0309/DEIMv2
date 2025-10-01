"""Teacher-student distillation utilities for DEIM models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


class TeacherStudentWrapper(nn.Module):
    """Wrap a student DEIM model with a frozen teacher for distillation.

    During training the wrapper runs the teacher in ``torch.no_grad`` mode and
    attaches ``teacher_logits`` / ``teacher_corners`` to the student outputs so
    the existing criterion can consume them. Only the student parameters require
    gradients; the teacher is kept frozen automatically.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        *,
        force_teacher_train_mode: bool = True,
        teacher_expects_targets: bool = True,
        propagate_aux_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.force_teacher_train_mode = force_teacher_train_mode
        self.teacher_expects_targets = teacher_expects_targets
        self.propagate_aux_outputs = propagate_aux_outputs

        # Freeze teacher parameters so optimizers ignore them.
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        if self.force_teacher_train_mode:
            self.teacher.train()
        else:
            self.teacher.eval()

        self._warned_missing_logits = False
        self._warned_missing_corners = False

    @contextmanager
    def _maybe_teacher_mode(self) -> Iterable[None]:
        """Temporarily switch the teacher to the requested mode."""
        prev_mode = self.teacher.training
        target_mode = self.force_teacher_train_mode
        if prev_mode != target_mode:
            self.teacher.train(target_mode)
        try:
            yield
        finally:
            if self.teacher.training != prev_mode:
                self.teacher.train(prev_mode)

    def forward(self, x: torch.Tensor, targets: Optional[Any] = None) -> Dict[str, Any]:  # type: ignore[override]
        outputs = self.student(x, targets=targets)

        if not self.training or self.teacher is None:
            return outputs

        if self.teacher_expects_targets and targets is None:
            return outputs

        with torch.no_grad():
            with self._maybe_teacher_mode():
                teacher_kwargs = {}
                if self.teacher_expects_targets:
                    teacher_kwargs["targets"] = targets
                teacher_outputs = self.teacher(x, **teacher_kwargs)

        self._attach_teacher_outputs(outputs, teacher_outputs)
        return outputs

    def _attach_teacher_outputs(self, outputs: Dict[str, Any], teacher_outputs: Dict[str, Any]) -> None:
        """Inject teacher predictions into the student output dictionary."""
        t_logits = teacher_outputs.get("pred_logits")
        t_corners = teacher_outputs.get("pred_corners")

        if t_logits is None:
            if not self._warned_missing_logits:
                print("[TeacherStudentWrapper] Teacher output lacks 'pred_logits'; skipping distillation logits.")
                self._warned_missing_logits = True
            return

        if t_corners is None:
            if not self._warned_missing_corners:
                print("[TeacherStudentWrapper] Teacher output lacks 'pred_corners'; skipping distillation corners.")
                self._warned_missing_corners = True
            return

        s_logits = outputs.get("pred_logits")
        s_corners = outputs.get("pred_corners")
        if s_logits is None or s_corners is None:
            return

        if s_logits.shape != t_logits.shape:
            raise ValueError(
                f"Teacher pred_logits shape {t_logits.shape} mismatches student {s_logits.shape}; "
                "ensure num_queries and class dimensions align."
            )
        if s_corners.shape != t_corners.shape:
            raise ValueError(
                f"Teacher pred_corners shape {t_corners.shape} mismatches student {s_corners.shape}; "
                "ensure num_queries and reg_max align."
            )

        outputs["teacher_logits"] = t_logits.detach()
        outputs["teacher_corners"] = t_corners.detach()

        if not self.propagate_aux_outputs:
            return

        student_aux = outputs.get("aux_outputs")
        teacher_aux = teacher_outputs.get("aux_outputs")
        if not student_aux or not teacher_aux:
            return

        count = min(len(student_aux), len(teacher_aux))
        for idx in range(count):
            s_aux = student_aux[idx]
            t_aux = teacher_aux[idx]
            if "pred_logits" in t_aux and "pred_corners" in t_aux:
                if s_aux.get("pred_logits") is not None and s_aux.get("pred_corners") is not None:
                    if s_aux["pred_logits"].shape == t_aux["pred_logits"].shape \
                            and s_aux["pred_corners"].shape == t_aux["pred_corners"].shape:
                        s_aux["teacher_logits"] = t_aux["pred_logits"].detach()
                        s_aux["teacher_corners"] = t_aux["pred_corners"].detach()

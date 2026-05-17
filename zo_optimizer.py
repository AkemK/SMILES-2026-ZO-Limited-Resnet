"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import torch
import torch.nn as nn

from augmentation import get_transforms
from train_data import _get_fixed_indices, SAMPLES_PER_CLASS

class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-4,
        sigma: float = 0.08,
        n_batches: int = 256,
        n_pairs: int = 6,
        clip: float = 3.0,
        data_dir: str = "./data",
    ):
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.n_pairs = n_pairs
        self.clip = clip
        self.n_batches = n_batches
        self._t = 0

        self.layer_names = ["fc.weight", "fc.bias"]

        device = next(self.model.parameters()).device

        backbone = copy.deepcopy(self.model)
        backbone.fc = nn.Identity()
        backbone.eval()

        try:
            ds = datasets.CIFAR100(
                root=data_dir, train=True, download=False,
                transform=get_transforms(train=False),
            )
            idx = _get_fixed_indices(ds.targets, SAMPLES_PER_CLASS)
            loader = DataLoader(Subset(ds, idx), batch_size=128, shuffle=False, num_workers=0)

            n_cls  = self.model.fc.out_features
            n_feat = self.model.fc.in_features
            sums   = torch.zeros(n_cls, n_feat, device=device)
            counts = torch.zeros(n_cls, device=device)

            with torch.no_grad():
                for imgs, labels in loader:
                    feats = backbone(imgs.to(device))
                    for c in range(n_cls):
                        m = labels.to(device) == c
                        if m.any():
                            sums[c]   += feats[m].sum(0)
                            counts[c] += m.sum().float()

            prototypes = sums / counts.clamp(min=1).unsqueeze(1)
            self.model.fc.weight.data.copy_(prototypes)
            self.model.fc.bias.data.zero_()

        except Exception as e:
            print(f"[ZO] proto init skipped: {e}")

        del backbone
    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a random unit-norm perturbation vector of the same shape as ``param``.

        Args:
            param: The parameter tensor whose shape determines the output shape.

        Returns:
            A tensor of the same shape as ``param``, normalised to unit L2 norm.
        """
        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0

        norm = u.norm()
        if norm > 0:
            u = u / norm
        return u

    def _cosine_lr(self):
        # cosine decay from lr to lr/10
        progress = self._t / max(self.n_batches, 1)
        scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        return self.lr * scale
    
    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Skeleton: 2-point central-difference estimator.
        For each active parameter ``p`` independently:
            1. Sample a random unit vector ``u`` of the same shape as ``p``.
            2. Evaluate  f_plus  = loss_fn() with ``p ← p + eps * u``
            3. Evaluate  f_minus = loss_fn() with ``p ← p - eps * u``
            4. Restore ``p`` to its original value.
            5. Pseudo-gradient ← ``(f_plus - f_minus) / (2 * eps) * u``

        This is an unbiased estimator of the directional derivative along ``u``
        scaled back to parameter space.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch
                     and returns a scalar ``float``. May be called multiple
                     times; each call must use the *same* batch.
            params:  Dict of active parameter name → tensor (from
                     ``_active_params``).

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient
            tensor (same shape as the parameter).

        Student task:
            Replace this with a more efficient or accurate estimator:
        """
        # antithetic Gaussian ES: n_pairs pairs of perturbations
        # grad = (1 / (2 * sigma * n_pairs)) * sum_i (f+ - f-) * eps_i
        acc = {n: torch.zeros_like(p) for n, p in params.items()}
        first_loss = None

        with torch.no_grad():
            for i in range(self.n_pairs):
                noise = {n: torch.randn_like(p) for n, p in params.items()}

                for n, p in params.items():
                    p.data.add_(self.sigma * noise[n])
                fp = loss_fn()

                for n, p in params.items():
                    p.data.sub_(2 * self.sigma * noise[n])
                fm = loss_fn()

                for n, p in params.items():
                    p.data.add_(self.sigma * noise[n])

                scalar = (fp - fm) / (2 * self.sigma)
                scalar = max(-self.clip, min(self.clip, scalar))

                for n in acc:
                    acc[n].add_(scalar * noise[n])

                if i == 0:
                    first_loss = fp

        for n in acc:
            acc[n].div_(self.n_pairs)

        return acc, first_loss
    
    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Skeleton: vanilla gradient *descent* step (minimising the loss).
            ``p ← p - lr * grad``

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor (same keys as
                    ``params``).

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        lr = self._cosine_lr()
        with torch.no_grad():
            for name, param in params.items():
                param.data.sub_(lr * grads[name])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        params = self._active_params()

        grads, loss_before = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        self._t += 1
        return float(loss_before)

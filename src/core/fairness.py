from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import GridSearch, DemographicParity
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import contextlib
import logging
from ..utils import setup_logger

class FairnessProcessor:
    def __init__(self, privileged_groups, unprivileged_groups):
        # Enhanced fairness processor with multiple mitigation techniques
        self.priv = privileged_groups
        self.unpriv = unprivileged_groups
        self.logger = setup_logger(__name__)
        self._session = None
        self._configure_tensorflow()

    def _configure_tensorflow(self):
        # Configure TensorFlow with version compatibility
        self.tf = tf
        if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
            self.tf.compat.v1.disable_eager_execution()

    @contextlib.contextmanager
    def active_session(self):
        # Context manager for TensorFlow session handling
        if self._session is None or getattr(self._session, '_closed', False):
            self._session = self.tf.compat.v1.Session() if hasattr(self.tf.compat, 'v1') else self.tf.Session()
        yield self._session

    def apply_reweighting(self, dataset):
        # Apply preprocessing reweighting
        return Reweighing(
            unprivileged_groups=self.unpriv,
            privileged_groups=self.priv
        ).fit_transform(dataset)

    def apply_adversarial_debiasing(self, dataset, num_epochs=50):
        # Apply in-processing adversarial debiasing
        start = datetime.now()
        with self.active_session() as sess:
            model = AdversarialDebiasing(
                privileged_groups=self.priv,
                unprivileged_groups=self.unpriv,
                scope_name='adv_debias',
                debias=True,
                num_epochs=num_epochs,
                sess=sess
            )
            model.fit(dataset)
        self.logger.info(f"Adversarial training completed in {(datetime.now()-start).total_seconds():.2f}s")
        return model

    def apply_postprocessing(self, estimator, X_train, y_train, sensitive_features):
        # Apply postprocessing fairness correction
        return ThresholdOptimizer(
            estimator=estimator,
            constraints="demographic_parity",
            predict_method='predict_proba'
        ).fit(X_train, y_train, sensitive_features=sensitive_features)

    def apply_inprocessing(self, estimator, X_train, y_train, sensitive_features):
        # Apply inprocessing fairness constraints
        return GridSearch(
            estimator=estimator,
            constraints=DemographicParity(),
            grid_size=10
        ).fit(X_train, y_train, sensitive_features=sensitive_features)

    def close(self):
        # Clean up resources
        if hasattr(self, '_session') and self._session is not None:
            self._session.close()
            self._session = None

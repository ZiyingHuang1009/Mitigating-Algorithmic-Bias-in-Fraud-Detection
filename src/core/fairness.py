from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf
from datetime import datetime
import numpy as np
from ..utils import setup_logger
import os
import contextlib

class FairnessProcessor:
    def __init__(self, privileged_groups, unprivileged_groups):
        # Initialize fairness processor with explicit validation
        if not privileged_groups or not unprivileged_groups:
            raise ValueError("Both privileged and unprivileged groups must be specified")
        
        self.priv = privileged_groups
        self.unpriv = unprivileged_groups
        self.logger = setup_logger(__name__)
        self._session = None
        
        # Configure TensorFlow with validation
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        # Configure TensorFlow with version checking
        self.tf = tf
        if not hasattr(tf, 'compat') or not hasattr(tf.compat, 'v1'):
            self.logger.warning("TensorFlow 1.x compatibility mode not available")
        else:
            self.tf.compat.v1.disable_eager_execution()
            
    @property
    def sess(self):
        # Lazy session creation with validation
        if self._session is None or (hasattr(self._session, '_closed') and self._session._closed):
            self.logger.info("Creating new TensorFlow session")
            if hasattr(self.tf.compat, 'v1'):
                self._session = self.tf.compat.v1.Session()
            else:
                self._session = self.tf.Session()
        return self._session
    
    @contextlib.contextmanager
    def active_session(self):
        # Context manager for session handling with validation
        session = self.sess
        if session is None:
            raise RuntimeError("TensorFlow session initialization failed")
        yield session

    def apply_reweighting(self, dataset):
        # Apply reweighting with input validation
        if not hasattr(dataset, 'protected_attributes'):
            raise ValueError("Dataset must have protected_attributes")
        
        if dataset.protected_attributes.size == 0:
            self.logger.warning("No protected attributes available for reweighting")
            return dataset
            
        # Check multiple protected attributes
        prot_attrs = [dataset.protected_attributes[:,i] 
                    for i in range(dataset.protected_attributes.shape[1])]
        
        valid_attrs = [attr for attr in prot_attrs 
                    if len(np.unique(attr)) >= 2]
        
        if not valid_attrs:
            self.logger.warning("No valid protected attributes for reweighting")
            return dataset
            
        # Apply reweighting using first valid attribute
        return Reweighing(
            unprivileged_groups=self.unpriv,
            privileged_groups=self.priv
        ).fit_transform(dataset)

    def apply_adversarial_debiasing(self, dataset, num_epochs=50):
        # Apply adversarial debiasing with explicit validation
        if not hasattr(dataset, 'features'):
            raise ValueError("Dataset must have features")
        
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        self.logger.info("Starting adversarial debiasing...")
        start = datetime.now()
        
        with self.active_session() as sess:
            model = AdversarialDebiasing(
                privileged_groups=self.priv,
                unprivileged_groups=self.unpriv,
                scope_name='adv_debias',
                debias=True,
                num_epochs=num_epochs,
                batch_size=128,
                classifier_num_hidden_units=32,
                sess=sess
            )
            model.fit(dataset)
        
        train_time = (datetime.now() - start).total_seconds()
        self.logger.info(f"Adversarial debiasing completed in {train_time:.2f}s")
        return model, train_time

    def close(self):
        # Clean up resources with validation
        if hasattr(self, '_session') and self._session is not None:
            if hasattr(self._session, 'close'):
                self._session.close()
            self._session = None
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf

def apply_reweighting(dataset, privileged, unprivileged):
    """Apply reweighting preprocessing"""
    return Reweighing(
        unprivileged_groups=unprivileged,
        privileged_groups=privileged
    ).fit_transform(dataset)

def apply_adversarial(dataset, privileged, unprivileged):
    """Apply adversarial debiasing"""
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        model = AdversarialDebiasing(
            privileged_groups=privileged,
            unprivileged_groups=unprivileged,
            scope_name='adv_debias',
            num_epochs=50,
            sess=sess
        )
        return model.fit(dataset)
Trained ASAHI models, one per chromosome: the models behind the paper's ENCODE results.

They were trained on the methylation atlas of Loyfer et al., at full depth.
No ENCODE library was used in training. A checkpoint is a dictionary of `hparams` and `state_dict`; load it with `asahi.model.load_checkpoint`.

"""``autoset``: the learned exposure and contrast of docs/18 (AS-04).

Dataset rows (``rows``), labels (``labels``), the joined training table (``dataset``), the ensemble and its file format
(``model``), training and k-fold evaluation (``train``, ``evaluation``), test vectors and the test model (``export``), the model
card (``model_card``) and synthetic rows for the tests (``synthetic``).

Run everything from the repository root as ``python3 -m training.autoset.<module>``. Nothing here ships in the app except the
model file and the vectors that ``crates/ym-color/tests/`` reproduces.
"""

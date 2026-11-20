"""Wan 2.2 TI2V-5B in MLX: the video model, with nothing but MLX and NumPy under it.

mflux has no video models, so Dreambot carries this one itself. It runs one set of weights:
FastMetal-5B-QAD, a three-step, 8-bit build of Alibaba's Wan 2.2 TI2V-5B made for Apple
Silicon by FastVideo (the Hao AI Lab at UCSD). Four parts, each a module here:

* :mod:`~dreambot.engine.wan.umt5` reads the prompt: Google's umT5-XXL encoder.
* :mod:`~dreambot.engine.wan.dit` is the model itself, a 5-billion-parameter transformer that
  turns noise into a video's latents under the prompt.
* :mod:`~dreambot.engine.wan.sampler` runs it for its three steps, and holds the first frame
  to a picture when a video is to start on one.
* :mod:`~dreambot.engine.wan.taehv` turns latents into frames, and a starting picture into a
  latent: TAEHV, a 22 MB stand-in for Wan's own 2.8 GB autoencoder.

A fifth, :mod:`~dreambot.engine.wan.look`, is Dreambot's own: it keeps a video in the colours
of the picture it starts on.

Everything is Apache 2.0 or MIT: Wan 2.2 and the FastMetal build (Apache 2.0), umT5 (Apache
2.0), TAEHV (MIT). The arithmetic follows FastVideo's MLX runtime (Apache 2.0), which is what
these weights were trained to run under, and is checked against it number for number in
``tests/test_wan.py``. FastVideo itself is not a dependency: it is a training framework with
PyTorch and fifty other packages under it, and all that is needed of it is this arithmetic.
``THIRD_PARTY.md`` at the top of the repository has both projects' notices.
"""

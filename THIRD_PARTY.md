# Third-party notices

Dreambot is MIT-licensed (see [LICENSE](LICENSE)). Its video engine, `src/dreambot/engine/wan`, is
Dreambot's own code, written against MLX and NumPy. It was written from two open-source projects, whose
arithmetic it follows so that the published weights give the results they were made to give. Their
notices are kept here. No code of either is installed with Dreambot or imported by it.

## FastVideo

<https://github.com/hao-ai-lab/FastVideo>, Apache License 2.0.

`engine/wan/dit.py` (the Wan 2.2 TI2V-5B transformer as it is read from a FastMetal checkpoint) and
`engine/wan/sampler.py` (the three-step schedule and the position tables) follow FastVideo's MLX runtime
(`fastvideo/mlx_runtime`) and its scheduler and rotary-embedding code. They are re-implementations, not
copies. What differs: nothing of PyTorch is used; the noise level is told per token through one code
path for videos from words and videos held to their first frames; more than one frame can be held; and
the weights are read straight from the checkpoint's two files.

The weights themselves (`FastVideo/FastMetal-5B-QAD` on Hugging Face) are a three-step, 8-bit build of
Alibaba's Wan 2.2 TI2V-5B, and both are Apache License 2.0. They are downloaded by
`dreambot models pull wan-2.2`, not distributed with Dreambot.

A copy of the Apache License 2.0 is at <https://www.apache.org/licenses/LICENSE-2.0>.

## TAEHV

<https://github.com/madebyollin/taehv>, MIT License.

`engine/wan/taehv.py` is a port of TAEHV's `taehv.py` to MLX: the same network, decoded a few frames at
a time. Its weights (`taew2_2.pth`) are downloaded from the author's repository by
`dreambot models pull wan-2.2`, not distributed with Dreambot.

```
MIT License

Copyright (c) 2025 Ollin Boer Bohan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## umT5

The text encoder (`engine/wan/umt5.py`) is Google's umT5-XXL, Apache License 2.0, read from the files
that come with the video model. The code is Dreambot's own, and is checked against the implementation in
Hugging Face's `transformers` (Apache License 2.0) in `tests/test_wan.py`.

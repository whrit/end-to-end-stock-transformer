# Third-Party Notices

## MASTER

This repository's MASTER backbone is adapted from the official
[SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER) implementation for:

> Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, and Sen Huang.
> “MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.”
> Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
> [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/27767)

Exact historical upstream commit was not recorded in the local research
snapshot. No commit SHA is inferred here.

The substantially similar adapted neural blocks are in
`src/market_guided_stock_alpha/modeling/master.py`: `PositionalEncoding`,
`SAttention`, `TAttention`, `Gate`, `TemporalAttention`, and `MASTER`.
`modeling/experimental/dual_frequency.py` reuses those adapted blocks inside a
project-specific exploratory gating architecture. The surrounding canonical
configuration, data transport, training, rolling, evaluation, feature/label,
ablation, baseline, and provenance systems are project-specific engineering.

The upstream snapshot retained during migration carried the following MIT
license notice. It is reproduced to preserve the upstream copyright and
permission notice for adapted portions:

A compact hash record of the deleted migration references is retained in
`results/provenance/upstream_master_reference.json`.

```text
MIT License

Copyright (c) 2025 Data Management Technology and AI

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

## Microsoft Qlib Alpha158

The standard Alpha158 feature family and fixed feature order implemented in
`src/market_guided_stock_alpha/features/alpha158.py` are a pandas/NumPy
adaptation of `Alpha158DL` in
[microsoft/qlib](https://github.com/microsoft/qlib). The source definition
recorded by the derived-data specification identifies
`qlib/contrib/data/loader.py`, class `Alpha158DL`, and source blob SHA
`4d11f3a34cc1d12728723956c8e9324556eee8e3`. That blob identifier documents
the feature-definition source used by this repository; it is not presented as
the exact runtime commit of every historical experiment.

Qlib is distributed under the MIT License:

```text
MIT License

Copyright (c) Microsoft Corporation.

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

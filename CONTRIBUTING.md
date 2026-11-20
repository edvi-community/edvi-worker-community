# Contributing

Open an issue to describe the problem you want to solve. Fork this repository
and submit a pull request against `main`. Explain what changed and how you
tested it. The maintainer reviews merges and selects community changes for
integration into official Edvi releases.

Install an editable copy with `python -m pip install -e .`, then run
`python -m unittest discover -s tests -v`. Tests use local loopback sockets
and mock model inference. If you test real inference, include the Mac model,
macOS version, and MLX and model revisions.

Use fictional credentials and inputs in tests. Keep model weights, installers,
databases, and local state out of Git. Report security issues privately using
the contact path in [SECURITY.md](SECURITY.md).

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Model and artifact trust

Core Archive V2/N4MM uses validated native formats. Historical Python `.n4a`
bundles and host-estimator artifacts can contain pickle/joblib payloads.
Loading these Python artifacts may execute code: only load artifacts from a
trusted producer and a trusted delivery channel. Checksums detect alteration;
they do not make an untrusted pickle safe. Never upload an unknown Python
artifact to a privileged process for inspection or prediction.

## Reporting a Vulnerability

If you discover a security vulnerability in nirs4all, please **do not open a public GitHub issue**.

Instead, report it privately via one of the following channels:

- **GitHub Security Advisories**: Use the "Report a vulnerability" button on the [Security tab](https://github.com/GBeurier/nirs4all/security/advisories/new) of this repository.
- **Email**: Contact the maintainer directly at [gregory.beurier@cirad.fr](mailto:gregory.beurier@cirad.fr) with the subject line `[SECURITY] nirs4all vulnerability`.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any suggested mitigations (if known)

We aim to acknowledge reports within **5 business days** and to provide a fix or mitigation within **30 days** for confirmed issues.

## Scope

This policy covers the `nirs4all` Python library published on PyPI.

Security issues in optional dependencies (TensorFlow, PyTorch, JAX, etc.) should be reported directly to those projects.

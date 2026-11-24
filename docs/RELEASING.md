# Releasing Vahana to PyPI

The wheels are built and checked by CI (`.github/workflows/wheels.yml`): one per platform, each installed into a clean environment and run through `vahana selftest` under every kernel build. A version tag publishes them, together with a source distribution, through PyPI trusted publishing. No API token is stored anywhere.

## One-time setup (the PyPI account owner)
1. Log in at https://pypi.org, open *Your account → Publishing*, and add a **pending publisher** (the project does not exist on PyPI yet):
   - PyPI project name: `vahana`
   - Owner: `vahana-ai` (the GitHub organization), repository: `Vahana`
   - Workflow name: `wheels.yml`
   - Environment name: `pypi`
2. On GitHub, open *Settings → Environments* and create an environment named `pypi`. Optionally require your own approval before each publish.

## Each release
1. Set the version in `pyproject.toml` and `src/vahana/__init__.py`, and add a `CHANGELOG.md` entry.
2. Commit, then tag and push:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
3. CI builds and checks every platform's wheel, builds the sdist, and the `publish` job uploads them. If any platform's selftest fails, nothing is published.

**Without trusted publishing:** download the `wheel-*` and `sdist` artifacts from the tag's CI run, then run `python -m twine upload dist/*` with your own PyPI token.

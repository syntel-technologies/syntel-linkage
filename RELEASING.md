# Releasing

Publishing is automated. A tag is the only trigger, and there is **no PyPI API token anywhere** —
not in a secret, not on a laptop. PyPI verifies the release workflow's OIDC identity per run, so
there is nothing to rotate and nothing to leak.

## One-time: register the trusted publisher

PyPI needs to be told, once, which workflow is allowed to publish this project. Until then the
`publish` job fails with `invalid-publisher: valid token, but no corresponding publisher` — the
token exchange working correctly and being refused, which is what it should do.

Because the project does not exist on PyPI yet, this is a **pending publisher**:

<https://pypi.org/manage/account/publishing/>

| Field | Value |
|---|---|
| PyPI Project Name | `syntel-linkage` |
| Owner | `syntel-technologies` |
| Repository name | `syntel-linkage` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name is not optional here: `release.yml` runs the publish job in an environment
called `pypi`, and PyPI matches on that claim. Leaving it blank makes the claims disagree and the
exchange fails in exactly the same way.

After the first successful publish the pending publisher becomes a normal one, attached to the
project.

## Optional: require a human before each publish

GitHub → Settings → Environments → `pypi` → **Required reviewers**. The publish job then waits,
and the approval is recorded against the release rather than living in somebody's memory.

## Every release after that

1. Bump `version` in `pyproject.toml` **and** `__version__` in `src/syntel_linkage/__init__.py`.
   The workflow refuses to publish unless they and the tag all agree — publishing from a branch
   would make "what is on PyPI" a question about timing.
2. Merge to `main`.
3. Tag and push:

```bash
git tag -a v0.2.0 -m "syntel-linkage 0.2.0" && git push origin v0.2.0
```

The workflow then verifies the version, runs CI on 3.13 and 3.14, builds, checks the metadata
renders on PyPI, publishes with attestations, and writes a GitHub release with the artefacts
attached.

## If a release fails

A version number is spent only on a **successful** upload — PyPI will not accept the same version
twice, even after deletion. A run that fails before publishing costs nothing: fix the cause and
re-run the failed job, or delete the tag and start again. A run that fails *after* uploading needs
a new version number.

# Independent release channels

Publishing a library package and publishing a product delivery artifact are
separate operations. Each channel retains its own source, artifact and runtime
gates. A failed image or installer does not undo or block a successful registry
publication.

## Python package

Complete the library scientific, type, metadata, installed-runtime and review
gates for the qualified source before pushing a publishing tag. An immutable
bare SemVer tag such as `1.6.1` starts `publish.yml`. The workflow
reproduces the qualified wheel, installs and checks that wheel, publishes to
PyPI, then verifies the public installed package. A GitHub Release, documentation
build, Docker image or desktop installer is not a prerequisite.

A manual registry retry runs from `main` with an existing reachable tag and
`publish_release=true`; this legacy input name now means registry publication.
The default manual run is a nonpublishing build. Publication still refuses an
already occupied PyPI version; retry failed post-publication checks directly
instead of trying to replace immutable package files.

## Docker image

Run `publish-docker.yml` independently from `main` with `release_tag` after the
qualified wheel is available on PyPI. Its preflight checks the tag and public
wheel hash. It builds and loads a local image, checks the native cohort and
package dependencies, then pushes that same qualified image. An image failure
can be corrected and retried without rebuilding or republishing the library.

Documentation has its existing independent `docs.yml` workflow. GitHub Releases
can describe a version and link completed channels; they do not initiate another
registry publication or imply that unfinished delivery artifacts are available.

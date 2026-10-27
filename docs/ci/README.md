# Optional GitHub Actions configuration

`verify.yml` runs the data audit and corruption tests on Python 3.11, without loading model weights.

The release uploader's GitHub authorization does not include the `workflow` scope, so this configuration is supplied as a template rather than an enabled workflow. To enable it later, use an appropriately authorized account to copy it to `.github/workflows/verify.yml` and commit the change. Repository creation and ordinary file pushes do not require that additional scope.

The release was checked locally and its committed file bytes were verified against the checksum manifest. No GitHub Actions success is claimed for this upload.

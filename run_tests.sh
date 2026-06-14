#!/bin/sh
set -e
black --check preframr_aug tests
pylint preframr_aug tests
pyright preframr_aug
pytest -n "${PYTEST_WORKERS:-auto}" --dist worksteal \
    --cov=preframr_aug --cov-report=term-missing --cov-fail-under=80

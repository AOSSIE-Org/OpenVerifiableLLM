# Network placement requests

Rental controller intent `ovl.rental-controller-intent.v4` adds required integer
`minDownload` and `minUpload` fields to the creation payload. Each must be between
1 and 100000. The exact values belong to the pinned intent and are passed unchanged
to RunPod's GraphQL `PodFindAndDeployOnDemandInput`.

These are provider placement requests, not measured throughput guarantees. Closure
reports list them under `provider_requested_only_fields`. Runtime downloads must
still finish within their selected deadlines and pass complete byte/hash checks;
qualification, production admission and replay remain separate requirements.

Version 4 retains the version 3 cloud selection and quote policy, the selected CUDA
versions, the original watchdog identity, all cost bounds and shutdown deadlines.
Older intent schemas reject these additional fields. No running intent is upgraded.

The provider field definitions are documented in the
[RunPod GraphQL specification](https://graphql-spec.runpod.io/#definition-PodFindAndDeployOnDemandInput).
REST v2 has a separate schema; this change does not add fields to REST requests.

Run `python -m pytest -q tests/test_rental_network_placement.py` for synthetic
request/restart/termination checks and adversarial pin, identity and value checks.
These tests create no paid resources and confer no training verification credit.

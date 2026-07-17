# PetCare DB V1 Demo Bundle

This folder is a ready-to-zip fixture bundle for the local agent harness.

Pet ids:

- `1`: Mochi, cat, cough scenario with normal recent baseline.
- `2`: Bori, dog, vomiting scenario with recent appetite/activity changes.
- `3`: Leo, cat, urinary concern scenario with urinary history.

First-time setup from the repository root:

```powershell
python -m pip install -e .
```

Run without zipping:

```powershell
python -m petcare_agent.harness --data-zip examples\data_bundles\petcare_db_v1_demo --pet-id 1
```

Run the zipped version:

```powershell
python -m petcare_agent.harness --data-zip examples\data_bundles\petcare_db_v1_demo.zip --pet-id 1
```

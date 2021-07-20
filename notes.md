
# TODO

Milestone 1. Minimally functional, vanilla

- Create HEAR API skeleton
- Add basic pytests using the HEAR API
- Implement the API using OpenL3 in simplest way possible
Scene embeddings: Average
- Setup Github Actions running pytest
- Make pip installable
- Also test pip install in Github Actions

# Misc
! openl3 on pypi released as 0.4.0 july 2021
But on github here only rc0 and rc1 tags
https://pypi.org/project/openl3/0.4.0/



The conflict is caused by:
    openl3 0.4.0 depends on h5py<3.0.0 and >=2.7.0
    tensorflow 2.5.0 depends on h5py~=3.1.0

Fixed by using Tensorflow 2.4.x


INFO: pip is looking at multiple versions of cython to determine which version is compatible with other requirements. This could take a while.
ERROR: Could not find a version that satisfies the requirement scikit-image<0.15.0,>=0.14.3 (from openl3) (from versions: 0.7.2, 0.8.0, 0.8.1, 0.8.2, 0.9.0, 0.9.1, 0.9.3, 0.10.0, 0.10.1, 0.11.2, 0.11.3, 0.12.0, 0.12.1, 0.12.2, 0.12.3, 0.13.0, 0.13.1, 0.14.0, 0.14.1, 0.14.2, 0.14.3, 0.14.5, 0.15.0, 0.16.2, 0.17.1, 0.17.2, 0.18.0rc0, 0.18.0rc1, 0.18.0rc2, 0.18.0, 0.18.1, 0.18.2rc1, 0.18.2rc2, 0.18.2)
ERROR: No matching distribution found for scikit-image<0.15.0,>=0.14.3


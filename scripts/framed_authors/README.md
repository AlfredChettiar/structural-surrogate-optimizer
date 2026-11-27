# FRAMED authors' preprocessing code

`PostSW.py` and `getInvalid.py` are copied from https://github.com/Lyleregenwetter/FRAMED (commit b881634, MIT licence, see LICENSE).
One change: `DataFrame.applymap` is renamed to `DataFrame.map` for pandas >= 3. Used only by `scripts/framed_gap_check.py`.

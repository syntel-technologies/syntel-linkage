# syntel-linkage

Probabilistic record linkage by the Fellegi-Sunter method, in the standard library.

## Install

```bash
pip install syntel-linkage
```

Python 3.13+. No dependencies.

## Usage

```python
from syntel_linkage import link

left = [
    {"unique_id": "L1", "postcode": "AD1 2XY", "surname": "haddad", "given": "layla"},
    {"unique_id": "L2", "postcode": "AD1 2XY", "surname": "nasser", "given": "omar"},
    {"unique_id": "L3", "postcode": "AD3 9QQ", "surname": "khoury", "given": "rami"},
]
right = [
    {"unique_id": "R1", "postcode": "AD1 2XY", "surname": "haddad", "given": "layla"},
    {"unique_id": "R2", "postcode": "AD1 2XY", "surname": "haddad", "given": "yusuf"},
    {"unique_id": "R3", "postcode": "AD3 9QQ", "surname": "khoury", "given": "rami"},
]

result = link(
    left,
    right,
    blocking_keys=["postcode"],
    comparison_fields=["surname", "given"],
    threshold=0.5,
)

for p in result.predictions:
    print(f"{p.left_id} ~ {p.right_id}  weight={p.weight:+.2f}  posterior={p.probability:.3f}")

print("m trained by:", result.training.method_m)
print("untrainable :", result.training.untrainable)
```

```text
L1 ~ R1  weight=+23.49  posterior=1.000
L3 ~ R3  weight=+23.49  posterior=1.000

m trained by: default (only 5 candidate pairs)
untrainable : ('postcode',)
```

Note the last two lines, which are the point of the library as much as the first two. Six records
do not contain enough evidence to learn m from, so it says so instead of returning a number that
looks trained. And `postcode` is a blocking key, so inside a block it always agrees and carries no
information — naming it is more useful than scoring it.

## Why this exists

Syntel's products need to decide whether a record in one system and a record in another describe
the same thing, and to say how confident that decision is with evidence a person can audit. The
established open-source implementation is Splink, which is MIT — but it requires `igraph`, which is
GPL-2.0, and Syntel's licence policy does not permit GPL in a service's runtime.

**No code was copied from Splink or from any other implementation.** Copying GPL code into a new
package does not change its licence; a derived work stays GPL, which would be the same risk with
extra steps. This is an independent implementation of the method published in Fellegi and Sunter,
*A Theory for Record Linkage*, Journal of the American Statistical Association 64 (1969), which is
mathematics rather than anyone's software.

## What it does

* **Blocking.** Compare only pairs that agree on a blocking key, so the comparison space stays
  bounded instead of quadratic.
* **u by random sampling.** The probability a field agrees between two records that are *not* a
  match, measured from randomly drawn pairs, which are almost all non-matches.
* **m by expectation maximisation.** The probability a field agrees between records that *are* a
  match, learned from the candidate pairs rather than asserted.
* **Match weight and posterior.** The log2 likelihood ratio per field, summed, plus the prior odds.
* **Clustering.** Connected components over accepted pairs, by union-find.

## What it deliberately does not do

* **Only exact agreement.** There are no fuzzy comparison levels: no Jaro-Winkler, no edit
  distance, no numeric tolerance. A caller that wants "similar" normalises before comparing and
  says which normalisation it used, so the transformation is recorded evidence rather than a
  hidden threshold.
* **No database backend.** Everything runs in this process over sequences the caller already has.
  That is a deliberate bound: this library is for adjudicating a candidate pair on a permitted
  sample, not for linking two warehouses.
* **No charts, no diagnostics, no model persistence.** The trained parameters come back as plain
  numbers for the caller to record with its own evidence.

## Honest reporting

Two situations produce a number that looks trained and is not, and both are reported rather than
hidden:

* **A field used for blocking cannot have its m probability trained.** Inside a block that field
  always agrees, so it carries no information. `TrainingReport.untrainable` names it.
* **Too few pairs to learn from.** Expectation maximisation on a handful of pairs returns whatever
  it started with. `TrainingReport.method` says `default` and why.

## Licence

Apache-2.0. See `LICENSE`.

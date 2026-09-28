# Data

## What is in here

`raw/cookie_cats.csv` holds 90,189 rows, one per player, from a live A/B test run
on the mobile puzzle game Cookie Cats by Tactile Entertainment. Players were
randomly assigned to one of two builds when they installed:

| column | meaning |
|---|---|
| `userid` | player identifier, unique across the file |
| `version` | `gate_30` when the first progression gate sat at level 30, `gate_40` when it was moved to level 40 |
| `sum_gamerounds` | game rounds played in the 14 days after install |
| `retention_1` | whether the player came back the day after installing |
| `retention_7` | whether the player came back seven days after installing |

The gate is a forced wait: progress stops until the player either waits out a
timer, asks a friend, or pays. Moving it later means more free play before the
first interruption, which is the change the experiment was testing.

## Where it came from

The dataset was published as the Mobile Games A/B Testing challenge and is
mirrored in several public repositories. This copy was taken from
<https://github.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats>,
file `cookie_cats.csv`, which is the version most widely used for teaching
experiment analysis. Credit for collecting and releasing the data belongs to
Tactile Entertainment and the DataCamp project that first packaged it.

## What is deliberately absent

There are no pre-assignment covariates: nothing about the player exists in the
file before the gate placement was decided. Every other column was measured in
the fourteen days after. That shapes what can honestly be done with the data and
is why `src/gatecheck/targeting.py` tests for a targeting signal rather than
assuming one, and why `src/gatecheck/validity.py` reports that covariate balance
cannot be checked at all.

## Generated files

`interim/players.parquet` is a cache written on the first run and ignored by git.
Delete it to force a re-read of the CSV.

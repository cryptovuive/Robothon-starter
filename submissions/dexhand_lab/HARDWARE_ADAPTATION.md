# DexHand Lab Hardware Adaptation Audit

This file documents a simulation-to-hardware replay audit. It is not a physical robot trial.

The command stream maps the simulated five-finger hand joints to LEAP/Shadow-style joint channels, clamps ranges, limits pressure targets, and includes emergency-stop checks for excessive slip or pressure.

---
name: Script Skill
version: 1.2.3
description: Script-backed fixture skill for testing skill wrappers.
metadata:
  skills:
    env:
      - FIXTURE_API_KEY
---

# Script Skill

## When to Use

Use this skill when you need to exercise bundled fixture scripts.

## Setup

Set `FIXTURE_API_KEY` before invoking the scripts.

## Usage

Run `scripts/echo_args.py` to capture arguments and stdin.

## Pitfalls

Do not invent script names.

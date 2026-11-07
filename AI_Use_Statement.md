# AI Use Statement

SE_POL 408 Final Project — Arraignment Bail and Remand Decisions in New York City

This file records how generative AI tools were used in this project. Every team member adds a row when they use an AI tool for project work. All analysis decisions, code and written conclusions were reviewed and are owned by the team. Per the course policy, each entry notes at least one AI output that had to be fixed or rejected.

## Summary

[To be completed before final submission (12/4): which tools were used, for what kinds of tasks, how outputs were verified, and a key example prompt.]

## Log

| Date | Member | Tool | Task | How the output was verified | What was fixed or rejected |
|---|---|---|---|---|---|
| 2026-11-06 | Yiling Long | Claude (Anthropic) | Drafted the topic-change plan (v1) and profiled the 2022 file | Rates recomputed from the raw file; legal and literature claims marked for verification before citing | — |
| 2026-11-07 | Yiling Long | Claude (Anthropic) | Revised the plan (v2), config, AGENTS.md, README and data notes after reading the UCS data dictionary and documentation | Each sample rule checked against the dictionary/documentation and value counts in the 2022 NYC file (case types, nominal bail share, arrest-cycle duplicates, court names) | Removed defense-counsel type from features (dictionary: most recent attorney, post-arraignment); dropped a planned community-court date filter because those courts are not listed separately in the NYC data; an AI-overwritten `.gitignore` was rewritten and tested |

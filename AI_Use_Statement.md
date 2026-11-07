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
| 2026-11-07 | Yiling Long | Claude (Anthropic) | Wrote `src/sample.py`, `src/build_dataset.py` and notebook 01 (sample rules, outcomes, anonymous judge codes, diagnostics) | Ran the full build on 2021–2023; checked counts against an independent 2022 query (117,294 dockets after S3), case_id uniqueness, outcome rates by year, and every category value against `value_counts()` | The first version cast `internal_case_id` to an integer and failed: the ID is a hexadecimal string, now kept as text. Missing-value handling for Y/N fields was fixed after the build failed on pandas' NA. The planned arrest-cycle rule turned out to matter little (60 rows) once superior-court cases were excluded; documented rather than dropped |
| 2026-11-07 | Yiling Long | Claude (Anthropic) | Added 2024–2025 data, checked them against 2021–2023, added `judge_id_final`, revised arrest-cycle de-duplication and memory use | Compared columns and decision categories across all years; compared the re-downloaded dictionary byte-for-byte with the earlier copy; re-ran the build and notebook 01 | Rejected an earlier AI-relayed claim that the official dictionary has a 'Nominal bail-set' category (the current dictionary does not). Changed de-duplication so a 2024 docket could no longer remove a 2023 docket from the sample. The notebook kernel ran out of memory; loading was changed to keep NYC rows only while reading |

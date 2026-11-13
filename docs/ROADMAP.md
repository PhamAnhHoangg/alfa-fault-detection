# Roadmap and future work

Last updated: 2026-11-13. This file lists what is done, what is queued, and what comes later.
No company data goes into this project.

## Done

- Merged documentation (`docs/ALFA_fault_detection_documentation.md`), pushed.
- Code index with script, function and line numbers (`docs/code_index.md`), pushed.
- Pictures 1 to 4 drawn (picture 4 is the script map, `diagram4_script_map.png`) and GIF 1 (time window, `gif1_time_window.gif`): time window, pipeline, split (`figures/diagram1_time_window.png`, `diagram2_pipeline.png`, `diagram3_split.png`, script `src/make_diagrams.py`). Pushed.
- Code review fixes (2026-11-13): pytest.ini saved without BOM so pytest starts, 09 creates `results/` before writing, 02 stops if a session lands in two groups, fixed titles removed from fig2 and fig8, uvicorn command in the docs gets `--app-dir src`, neutral setup paths in the docs, three new lines under "Limits and open questions".

## Queue: next steps

### A. Pictures and visuals
1. Done: pictures 1 to 4 and the fig GIFs are pushed.
2. Picture 4: fig7 cell as a scatter (each engine row as a dot, x~ against y~, with the line).
3. Picture 5: raw signals by fault type around the fault, one panel per type.
4. Picture 6: detector flow, A, F and D2 side by side with what each reads.
5. Turn every picture into a step-by-step GIF with a caption on each step (script `src/make_gifs.py`). Done: GIF 1 time window, GIF 2 pipeline, GIF 3 split (with real split tables), GIF 4 script map, GIF 5 where the numbers come from (real rows of one engine-fault flight), GIF 6 the fig7 cell 0.65, and `walkthrough_numbers.mp4` (GIF 5 and 6 joined). Data snippets for them are in `docs/gif_data/` (made by `src/export_gif_data.py`, drawn by `src/make_gifs_data.py`). To do: the other 27 fig7 cells, std5 and delta10 traces over a whole flight. To do: Then possibly one short video that joins them, linked from the README.
6. Fix the pipeline-map widget (fig8 reads `model_comparison.csv`, and the right panel shows engine-fault recall) and animation steps 5 and 6 (pooling happens after per-flight demeaning).

### B. README
7. Done: the README Summary and Results table already cover it.
8. Done: "How to read this repo" section added under the Summary.
9. Done: "Related work" section added to the README (ALFA paper, ICRA 2019 paper, alfa-dataset-tools, DronLomaly).
10. Licence: undecided.

### C. Documentation
11. Done for the setup steps. The experiments scripts still hold the original paths.
12. Fix the old `ALFA_DATA` wording in the three original Claude docs (or treat the merged file as the main copy).
13. Symbols and numbers table: symbol, meaning, number, where it is in the data.
14. Code explanations: line-by-line tables, and a task list per Python file (reads, does, writes).
15. Formulas written like in a thesis (test real equations in the doc, otherwise use the PDF route).
16. "Unsolved problems" section.
17. Dropped on 2026-11-13: no "Mistaken concepts" section.
18. PDF of the merged document with the figure index and issue list at the top.
19. PowerPoint deck (new or updated, no company data).

### D. Learning path
20. Quizzes stopped on 2026-11-13 after 08 and 09.

### E. Open checks
21. Start-of-flight drift check: does drift at the start of a flight drive the engine correlations in fig7?
22. Same step-by-step trace for the `std5` and `delta10` rows (the `mean5` cells are already traced).
23. Done: fig9_calibration.png committed on 2026-11-13.
24. Done: key point 2 now says a feature at second t uses readings up to the end of second t (01_build_table.py line 61).
25. Done: the fig2 title now uses only the flight name. Regenerate fig2 and fig8 on the PC.
26. Done.
31. Done. Check whether sessions that start seconds apart (for example 2018-09-11-14-41-38 in cv3 and 2018-09-11-14-41-51 in test) overlap in time. Script: `src/20_session_overlap.py`. Result on 2026-11-13: 6 overlaps. The test pair (14-41-51 test, 14-41-38 cv3) shares no timestamps and overlaps about 6 s at the edge; 14-41-38 is a no-failure flight. The 2018-10-05 pair and the 2018-10-18 group were checked the same day: no shared timestamps and no altitude trace that matches another at any shift, so no copied flights. Split kept as it is.
32. Decide whether to redo the signal screening (step6 to step8) on dev flights only.

## Later: bigger projects

27. Own small MCP server that loads the trained model and scores a flight. No ALFA MCP server was found in a search on 2026-11-10, so this would be new. Build step by step after the docs.
28. Extra test of the detector on a large public PX4 flight-log dataset (1396 real logs reported in a paper; title and link still to find). Needs a mapping from its signals to the 21 features.
29. Optionally publish the model and the per-second table on Hugging Face. Check the ALFA licence first.
30. Read related work found in the search: castacks/alfa-dataset-tools, DronLomaly, Flogtool, the ardupilot-mcp log diagnosis server.

## Rules that stay fixed

- The test set is locked and scored once.
- No API keys in chat or repo.
- No company data in docs, slides or code.
- Commits carry no Co-Authored-By or session trailers. The only Claude credit is the line "AI assistance tool: Claude Sonnet 5.5 (Anthropic)" at the end of the README Credits.

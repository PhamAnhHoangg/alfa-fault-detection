# Roadmap and future work

Last updated: 2026-11-10. This file lists what is done, what is queued, and what comes later.
No company data goes into this project.

## Done

- Merged documentation (`docs/ALFA_fault_detection_documentation.md`), pushed.
- Code index with script, function and line numbers (`docs/code_index.md`), pushed.
- Pictures 1 to 4 drawn (picture 4 is the script map, `diagram4_script_map.png`) and GIF 1 (time window, `gif1_time_window.gif`): time window, pipeline, split (`figures/diagram1_time_window.png`, `diagram2_pipeline.png`, `diagram3_split.png`, script `src/make_diagrams.py`). Not pushed yet.

## Queue: next steps

### A. Pictures and visuals
1. Push pictures 1 to 3 and place them in the docs and README.
2. Picture 4: fig7 cell as a scatter (each engine row as a dot, x~ against y~, with the line).
3. Picture 5: raw signals by fault type around the fault, one panel per type.
4. Picture 6: detector flow, A, F and D2 side by side with what each reads.
5. Turn every picture into a step-by-step GIF with a caption on each step (script `src/make_gifs.py`). Done: GIF 1 time window, GIF 2 pipeline, GIF 3 split (with real split tables), GIF 4 script map, GIF 5 where the numbers come from (real rows of one engine-fault flight), GIF 6 the fig7 cell 0.65, and `walkthrough_numbers.mp4` (GIF 5 and 6 joined). Data snippets for them are in `docs/gif_data/` (made by `src/export_gif_data.py`, drawn by `src/make_gifs_data.py`). To do: GIF for fig1 to fig5 and fig8 (highlight parts of the real plots), the other 27 fig7 cells, std5 and delta10 traces over a whole flight. To do: Then possibly one short video that joins them, linked from the README.
6. Fix the pipeline-map widget (fig8 reads `model_comparison.csv`, and the right panel shows engine-fault recall) and animation steps 5 and 6 (pooling happens after per-flight demeaning).

### B. README
7. "Results at a glance" table at the top.
8. "How to read this repo" section pointing to the contents list.
9. "Related work" section citing the ALFA paper (Keipour, Mousaei, Scherer, IJRR 2021, arXiv 1907.06268) and DronLomaly (LSTM log anomaly detection).
10. Add an MIT LICENSE file.

### C. Documentation
11. Neutral paths in the docs (for example `X:\your-folder`) instead of the local drive paths.
12. Fix the old `ALFA_DATA` wording in the three original Claude docs (or treat the merged file as the main copy).
13. Symbols and numbers table: symbol, meaning, number, where it is in the data.
14. Code explanations: line-by-line tables, and a task list per Python file (reads, does, writes).
15. Formulas written like in a thesis (test real equations in the doc, otherwise use the PDF route).
16. "Unsolved problems" section.
17. "Mistaken concepts" section, grown through quizzes.
18. PDF of the merged document with the figure index and issue list at the top.
19. PowerPoint deck (new or updated, no company data).

### D. Learning path
20. Continue teaching the scripts step by step: next `08_correlation.py`, then `09` onward.

### E. Open checks
21. Start-of-flight drift check: does drift at the start of a flight drive the engine correlations in fig7?
22. Same step-by-step trace for the `std5` and `delta10` rows (the `mean5` cells are already traced).
23. Decide whether fig9 is committed or stays a text note.
24. Re-check the wording of key point 2 ("never after").
25. Fix the fig2 title hard-coded in `05_figures.py` (it may not match the first rudder test flight).
26. Make the error message in `02_make_split.py` neutral.

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

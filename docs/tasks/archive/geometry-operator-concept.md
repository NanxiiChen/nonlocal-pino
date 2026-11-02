# Simplify the geometry-operator explanation

User says the detailed HTML is too complex and requests a conceptual explanation.
Deliver a short illustrated version: varying geometry -> two point maps ->
conditioned position/time query. Keep real geometry/initial-point data; omit
tensor dimensions, formulas, optimizer/attention/kernel details. Preserve the
previous detailed artifact as optional background. No model or training change.

Status COMPLETE. Short illustrated three-step page uses actual0/256/768geometry
point maps. Distinguishes condition (which domain/initial state) from query
(where/when); no formulas, dimensions, optimizer or attention implementation.
Three geometry switches and desktop/390pxmobile overflow checks pass, no JS
errors. Preview visually inspected. Detailed version retained through an optional
link. No model/training/Git change, blocker or queued follow-up.

HTML: /root/nonlocal-pinn/docs/tasks/geometry-operator-concept.html, copied to the
same thread visualization folder as the technical version. Template: sibling
geometry-operator-concept.html.in; generator: geometry_operator_concept.py.
Regenerate the detailed page first to provide its verified real input data,
then run the conceptual generator. Receipt: geometry-operator-concept-validation.json;
preview: geometry-operator-concept-preview.png. Generated HTML/PNG ignored.
Browser initially used an ambiguous test-only selector matching main+button;
narrowing tobuttonpassed without changing the page. Memory updated: PROGRESS,
DECISIONS and this archive. No remaining work for this explanation.

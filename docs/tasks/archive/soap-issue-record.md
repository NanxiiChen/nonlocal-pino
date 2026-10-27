# Record deferred SOAP issue

Goal: create a standalone issue with evidence, limits and a future resume point;
mark optimizer stability work deferred at the user's explicit request.

Acceptance: separate Markdown issue linked from progress/example docs, existing
records preserved, no optimizer/default/dependency/training changes.

Status: complete (2026-10-27). Blocker: none.

Changes: created docs/issues/soap-repeatability.md with symptoms, audit limits,
evidence links and an explicit user-requested deferral. Linked it from the example
README and PROGRESS; updated the root README structure note and appended the
decision to DECISIONS. No optimizer, default, dependency or training changes.

Validation: local issue links and the new progress/example links checked;
git diff --check passed. Documentation only; no numerical tests needed.

Next action: none. Only resume the deferred numerical investigation on explicit
user request, starting from docs/issues/soap-repeatability.md.

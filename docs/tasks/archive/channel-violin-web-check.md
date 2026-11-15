# Verify the reported stale website distribution

## Goal / context

User reports the website violin may not have updated. Latest approved release:
content0d831f1/mainf2b9127, verified Pages43556036346; latest metadata mainbd36c67.
The selected violin PNG is117510bytes, PDF48420bytes. Website source references
`assets/channel_nsch2d_fem_distribution.png` in the two-phase-flow chapter.
Prior publication check compared all13live resources successfully.

## Acceptance / status / next action

Completed. At2026-11-14T04:24:17UTC ordinary and cache-bypassing requests for
HTML/PNG/PDF all returned200 and matched selected files byte-for-byte. HTML
references the correct violin file/alt text. All responses advertise
Cache-Control:max-age=600. A fresh Chromium context loads the actual live page
and displays the expected2007x637violin; screenshot visually inspected.

A stale user-side cache is plausible but not directly observed. Provide the
direct PNG URL with `?v=20261114-violin` and force-refresh instructions. No server
content defect or need to redraw/redeploy was found. Do not claim a cache fix
was deployed. No Git action, training or source/asset change was performed.

Receipt: `artifacts/channel_violin_web_validation.json` (six HTTP checks and
fresh-browser result); screenshot `/tmp/channel-violin-live-browser.png`.
Memory changes: PROGRESS,DECISIONS,this archive. All unrelated work preserved.
No pending job or blocker. If the user still reports a different image, request
the exact page address to distinguish another case or an outdated URL.

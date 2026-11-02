"""Offline Chromium interaction and responsive-layout checks for the explainer."""
import json
import os
from pathlib import Path
import sys

# Existing isolated rendering tools; no training environment dependency change.
sys.path.insert(0, '/tmp/fracture2d-pdf-tools')
from playwright.sync_api import sync_playwright

REPO = Path('/root/nonlocal-pinn')
HTML = REPO/'docs/tasks/geometry-operator-explainer.html'
CHROME = '/tmp/fracture2d-pdf-browsers/chromium-1243/chrome-linux64/chrome'


def main():
    failures, requests = [], []
    screenshots = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage'],
            env=dict(os.environ, LD_LIBRARY_PATH='/tmp/fracture2d-pdf-libs/root/usr/lib/x86_64-linux-gnu',
                     FONTCONFIG_FILE='/tmp/fracture2d-fontconfig.xml'))
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda e: failures.append(str(e)))
        page.on('request', lambda r: requests.append(r.url))
        page.goto(HTML.as_uri(), wait_until='load')
        page.locator('#geometry-operator-explainer[data-ready=true]').wait_for()
        records = []
        for case in range(3):
            page.locator(f'[data-case="{case}"]').click()
            expected = page.evaluate('window.geometryExplainer.data.cases[window.geometryExplainer.state.sample].nodes')
            assert page.locator('#node-count').inner_text() == str(expected)
            assert page.locator('#geo circle[data-stream=boundary]').count() == 192
            assert page.locator('#geo circle[data-stream=domain]').count() == expected
            page.locator('#geo circle[data-stream=boundary][data-index="96"]').click()
            assert '孔壁' in page.locator('#inspector').inner_text()
            records.append({'case': case, 'nodes': expected, 'boundary_tokens': 192})
        page.locator('#normal-toggle').click()
        assert page.locator('#geo line[marker-end]').count() == 24
        page.locator('[data-view=domain]').click()
        assert page.locator('#geo circle[data-stream=boundary]').count() == 0
        page.locator('#geo circle[data-stream=domain]').nth(100).click()
        assert 'φ₀' in page.locator('#inspector').inner_text()
        page.locator('#edge-toggle').click()
        assert '图中最多显示 24 条' in page.locator('#inspector').inner_text()
        page.locator('#split-toggle').evaluate('(el)=>el.closest("details").open=true')
        before = page.locator('#split-value').inner_text()
        page.locator('#split-toggle').click()
        assert page.locator('#split-value').inner_text() == before
        assert page.locator('#split-label').inner_text() == '4 行 token'
        page.locator('#time-slider').evaluate('(el)=>{el.value=3;el.dispatchEvent(new Event("input",{bubbles:true}));}')
        assert page.locator('#time-value').inner_text() == '3.00'
        assert '1.000' in page.locator('#time-equation').inner_text()
        page.locator('#pool-step').click()
        assert page.locator('#pool-grid .fresh').count() == 32
        page.locator('#pool-step').click()
        assert page.locator('#pool-grid .fresh').count() == 64
        page.locator('#pool-reset').click()
        assert page.locator('#pool-grid .fresh').count() == 0
        page.locator('[data-case="0"]').click()
        page.locator('#geometry').scroll_into_view_if_needed()
        shots = REPO/'docs/tasks'
        screenshots['desktop'] = str(shots/'geometry-operator-explainer-desktop.png')
        page.screenshot(path=screenshots['desktop'])
        screenshots['network'] = str(shots/'geometry-operator-explainer-network.png')
        page.locator('#network').screenshot(path=screenshots['network'])
        page.set_viewport_size({'width': 390, 'height': 844})
        page.goto(HTML.as_uri(), wait_until='load')
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile horizontal overflow'
        page.locator('[data-case="1"]').click()
        assert page.locator('#node-count').inner_text() == '1229'
        screenshots['mobile'] = str(shots/'geometry-operator-explainer-mobile.png')
        page.screenshot(path=screenshots['mobile'], full_page=True)
        assert not failures, failures
        assert all(url.startswith('file:') for url in requests), requests
        browser.close()
    receipt_path = REPO/'docs/tasks/geometry-operator-explainer-validation.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.update(status='passed', browser_validation=dict(status='passed',
        engine='Existing isolated Chromium1243/Playwright', real_cases=records,
        normal_arrows=24, token_clicks=True, view_switches=True, scalar_token_split_invariant=True,
        time_slider=True, online_pool_controls=True, mobile_width=390, horizontal_overflow=False,
        javascript_errors=failures, external_requests=0), screenshots=screenshots)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(receipt['browser_validation'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

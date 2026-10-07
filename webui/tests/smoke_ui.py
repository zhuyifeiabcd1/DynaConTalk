"""Headless interaction smoke test for DynaConTalk Studio.

Usage:  python webui/tests/smoke_ui.py <base_url> <generation job id> <edit job id>

Drives the page with Playwright (system Chrome) on a finished generation job and a finished
edit job of it, exercises the main flows and writes screenshots to outputs/studio_smoke/.
It never launches a GPU job: the edit job it submits is cancelled right after it is queued.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE, JOB, EDIT_JOB = sys.argv[1:4]
OUT = Path(__file__).resolve().parents[2] / "outputs" / "studio_smoke"
OUT.mkdir(parents=True, exist_ok=True)
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.add_init_script("try { localStorage.setItem('studio.lang', 'en'); } catch (e) {}")
    errors: list[str] = []
    page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.goto(f"{BASE}/#job={JOB}")
    page.wait_for_selector(".job-row", timeout=15000)
    page.wait_for_timeout(1500)

    # ---- layout sanity
    card = page.evaluate("() => { const c = document.querySelector('#libGrid .card'); return c ? c.getBoundingClientRect().height : 0; }")
    check(card > 80, f"library cards have a real height ({card:.0f}px)")
    n_cards = page.locator("#libGrid .card").count()
    check(n_cards >= 150, f"library lists keyposes ({n_cards})")
    kp_cats = page.evaluate("""() => {
        const c = {};
        for (const x of window.__studio.state.assets.keyposes) c[x.category] = (c[x.category] || 0) + 1;
        return c; }""")
    expressive = ["wide", "head", "reach", "behind", "fold", "lean"]
    missing = [c for c in expressive if not kp_cats.get(c)]
    check(not missing, f"expressive keypose categories are populated ({ {c: kp_cats.get(c) for c in expressive} })")
    covers = page.evaluate("""async () => {
        const urls = window.__studio.state.assets.keyposes.slice(0, 12).map(x => x.media_url);
        const sizes = await Promise.all(urls.map(u => new Promise(res => {
            const im = new Image(); im.onload = () => res(im.naturalWidth / im.naturalHeight);
            im.onerror = () => res(0); im.src = u; })));
        return sizes; }""")
    check(all(abs(r - 4 / 3) < 0.02 for r in covers if r),
          f"keypose covers share one 4:3 framing ({[round(r, 2) for r in covers][:4]})")
    check(page.locator("#trackChunks .chunk-block").count() >= 1, "timeline shows chunk blocks")
    check(page.locator("#tlRuler .tick").count() > 5, "ruler has ticks")
    page.screenshot(path=str(OUT / "01_job.png"))

    # ---- words lane + captions (the test job has features/words.json)
    words_loaded = page.evaluate("() => !!(window.__studio && window.__studio.state.words)")
    if words_loaded:
        check(page.locator("#trackWords .word, #trackWords .sentence").count() > 0, "words lane renders word blocks")
        # step the playhead into the first word with the keyboard (Shift+Right = 10 frames)
        w0 = page.evaluate("() => window.__studio.state.words.words[0]")
        target = (int(w0["sf"]) + int(w0["ef"])) // 2
        page.keyboard.press("Home")
        for _ in range(target // 10):
            page.keyboard.press("Shift+ArrowRight")
        for _ in range(target % 10):
            page.keyboard.press("ArrowRight")
        page.wait_for_timeout(200)
        check(len(page.evaluate("() => document.querySelector('#captions').textContent")) > 0, "captions show the current sentence")
        check(page.locator("#trackWords .cur").count() == 1, "current word is highlighted in the lane")
        page.locator("#btnCaptions").click()
        page.wait_for_timeout(100)
        check(page.evaluate("() => document.querySelector('#captions').textContent") == "", "CC toggle hides captions")
        page.locator("#btnCaptions").click()
    else:
        print("SKIP words lane checks (no words.json for the test job)")

    # ---- library: select a keypose, open detail, insert with K
    page.locator("#libGrid .card").nth(3).click()
    page.wait_for_timeout(200)
    check(page.locator("#libFooter.on").count() == 1, "asset detail panel opens on card click")
    page.keyboard.press("End")
    page.keyboard.press("Home")
    page.keyboard.press("ArrowRight")
    for _ in range(5):
        page.keyboard.press("Shift+ArrowRight")
    page.wait_for_timeout(150)
    frame = page.evaluate("() => document.querySelector('#tcFrame').textContent")
    check(frame.strip() == "f 51", f"frame stepping moved the playhead to f51 (got '{frame}')")
    page.keyboard.press("k")
    page.wait_for_timeout(300)
    check(page.locator("#trackKp .kp-marker:not(.committed)").count() == 1, "K inserts a pending keypose marker")
    check(page.locator("#inspEditBody .insp-card.kp").count() == 1, "inspector shows the keypose editor")
    if words_loaded:
        check(page.locator("#inspEditBody .word-here").count() == 1, "keypose editor shows the word at that frame")
    page.screenshot(path=str(OUT / "02_keypose.png"))

    # ---- change part + sigma through the inspector
    page.locator("#inspEditBody .insp-card.kp select").first.select_option("full_body")
    page.wait_for_timeout(200)
    part = page.evaluate("() => JSON.parse(localStorage.getItem('studio.pending.' + '%s')).keyposes[0].part" % JOB)
    check(part == "full_body", "part change is persisted")

    # ---- trajectory: T adds a move at the playhead, edit its type
    page.keyboard.press("Home")
    for _ in range(30):
        page.keyboard.press("Shift+ArrowRight")
    page.keyboard.press("t")
    page.wait_for_timeout(400)
    check(page.locator("#trackTr .tr-seg:not(.committed)").count() == 1, "T adds a trajectory segment")
    check(page.locator("#inspEditBody .insp-card.tr").count() == 1, "inspector shows the segment editor")
    page.locator("#inspEditBody .prim-grid button", has_text="Crouch").click()
    page.wait_for_timeout(600)
    seg = page.evaluate("() => JSON.parse(localStorage.getItem('studio.pending.' + '%s')).traj[0]" % JOB)
    check(seg["type"] == "crouch", f"segment type switched to crouch ({seg['type']})")
    pv = page.evaluate("() => window.__studioPreview || null")
    page.screenshot(path=str(OUT / "03_traj.png"))

    # ---- drag the segment on the timeline
    box = page.locator("#trackTr .tr-seg:not(.committed)").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 120, box["y"] + box["height"] / 2, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    seg2 = page.evaluate("() => JSON.parse(localStorage.getItem('studio.pending.' + '%s')).traj[0]" % JOB)
    check(seg2["start"] > seg["start"], f"dragging moved the segment ({seg['start']} -> {seg2['start']})")

    # ---- pending badge + apply enabled
    check(page.locator("#tlPending .n").count() == 2, "pending badge shows both edit kinds")
    check(page.evaluate("() => !document.querySelector('#btnApply').disabled"), "Apply is enabled with pending edits")

    # ---- submit an edit job (queues; the GPU guard keeps it waiting), then cancel + delete it
    page.locator("#btnApply").click()
    page.wait_for_timeout(2500)
    new_id = page.evaluate("() => location.hash.replace('#job=', '')")
    check(new_id != JOB and len(new_id) > 10, f"edit job created and selected ({new_id})")
    detail = page.evaluate("(id) => fetch('/api/jobs/' + id).then(r => r.json())", new_id)
    check(detail["state"] in {"queued", "waiting", "running"}, f"new job state is {detail['state']}")
    check(len(detail["keyposes"]) == 1 and len(detail["traj_script"]) == 1, "request carries keyposes + traj_script")
    check(detail["keyposes"][0]["part"] == "full_body" and detail["traj_script"][0]["type"] == "crouch",
          f"edit payload preserved editor values (kp={detail['keyposes'][0]} tr={detail['traj_script'][0]})")
    page.screenshot(path=str(OUT / "04_queued.png"))
    page.evaluate("(id) => fetch('/api/jobs/' + id + '/cancel', {method: 'POST'})", new_id)
    page.wait_for_timeout(1500)
    st = page.evaluate("(id) => fetch('/api/jobs/' + id).then(r => r.json()).then(j => j.state)", new_id)
    check(st == "cancelled", f"cancel works ({st})")
    page.evaluate("(id) => fetch('/api/jobs/' + id + '?purge=1', {method: 'DELETE'})", new_id)
    page.wait_for_timeout(800)

    # ---- generate tab, language toggle, help modal
    page.locator("#inspTabs button[data-tab=generate]").click()
    page.wait_for_timeout(200)
    check(page.locator("#inspGenerate.on").count() == 1, "generate tab opens")
    page.locator("#langToggle button[data-lang=en]").click()
    page.wait_for_timeout(200)
    check(page.evaluate("() => document.querySelector('#btnGenerate span').textContent") == "Generate motion", "language toggle to EN")
    page.locator("#langToggle button[data-lang=zh]").click()
    page.keyboard.press("?")
    page.wait_for_timeout(200)
    check(page.locator("#modal:not(.hidden)").count() == 1, "help modal opens with ?")
    page.keyboard.press("Escape")
    page.locator("#libraryTabs button[data-tab=trajectories]").click()
    page.wait_for_timeout(300)
    check(page.locator("#libGrid .card.traj").count() >= 20, "trajectory library renders")

    # --- trajectory taxonomy: amplitude groups, frequency filter, quiet default
    measure = """() => [...document.querySelectorAll('#libraryTabs button span')]
        .map(s => ({t: s.textContent, slack: s.clientWidth - s.scrollWidth}))"""
    tab_report = {}
    for lang in ("en", "zh"):
        page.locator(f"#langToggle button[data-lang={lang}]").click()
        page.wait_for_timeout(200)
        tab_report[lang] = page.evaluate(measure)
    ok = all(len(v) == 3 and all(x["slack"] >= 0 for x in v) for v in tab_report.values())
    check(ok, "all three library tabs render un-clipped in both languages "
              f"({ {k: [(x['t'], x['slack']) for x in v] for k, v in tab_report.items()} })")
    groups = page.evaluate("() => [...document.querySelectorAll('#libGrid .lib-group')].map(g => g.firstChild.textContent)")
    check(len(groups) >= 4, f"trajectories are grouped by amplitude ({groups})")
    freq_chips = page.locator("#libChips .chips.sub .chip-btn")
    check(freq_chips.count() >= 2, "frequency filter row is present")
    total = page.locator("#libGrid .card.traj").count()
    freq_chips.first.click()
    page.wait_for_timeout(250)
    filtered = page.locator("#libGrid .card.traj").count()
    check(0 < filtered < total, f"frequency chip filters the grid ({total} -> {filtered})")
    freq_chips.first.click()
    page.wait_for_timeout(250)
    default_id = page.evaluate("() => window.__studio.state.config.default_trajectory")
    check(page.evaluate("() => document.querySelector('#genTrajectory').value") == default_id,
          f"generate form starts on the quiet default ({default_id})")
    calm = page.evaluate("""() => {
        const a = window.__studio.state.assets.trajectories.find(x => x.is_default);
        return a ? a.motion.travel_per_sec : null; }""")
    check(calm is not None and calm < 0.03, f"the default trajectory barely moves ({calm} m/s)")

    first_id = page.locator("#libGrid .card.traj").first.get_attribute("data-id")
    page.locator("#libGrid .card.traj").first.dblclick()
    page.wait_for_timeout(300)
    check(page.evaluate("() => document.querySelector('#genTrajectory').value") == first_id,
          "double-click sets the base trajectory")
    page.screenshot(path=str(OUT / "05_generate.png"))

    # ---- A/B toggle on an edit job with a source video
    page.goto(f"{BASE}/#job={EDIT_JOB}")
    page.wait_for_timeout(2000)
    check(page.locator("#abToggle:not(.hidden)").count() == 1, "A/B toggle visible for an edited chunk")
    src_before = page.evaluate("() => document.querySelector('#video').getAttribute('src')")
    page.locator("#abToggle button[data-ab=source]").click()
    page.wait_for_timeout(500)
    src_after = page.evaluate("() => document.querySelector('#video').getAttribute('src')")
    check(src_before != src_after, "A/B toggle swaps the video source")
    page.screenshot(path=str(OUT / "06_ab.png"))

    print("console errors:", errors[:10])
    check(not errors, "no console errors")
    browser.close()

print("\nRESULT:", "OK" if not failures else f"{len(failures)} failure(s)")
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)

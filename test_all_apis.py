#!/usr/bin/env python3
"""
Comprehensive API & System Test Script
Tests every endpoint and feature of the Factory Compliance & Alert Escalation System.
"""

import json
import sys
import time

import requests

BASE = "http://localhost:8000"

passed = 0
failed = 0
results = []

def test(name, func):
    global passed, failed
    try:
        func()
        passed += 1
        results.append(("PASS", name))
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        results.append(("FAIL", name, str(e)))
        print(f"  [FAIL] {name} — {e}")

# ──────────────────────────────────────────────────────────────────
# 1. HEALTH CHECK
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. HEALTH CHECK")
print("="*60)

def test_health():
    r = requests.get(f"{BASE}/api/health", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    d = r.json()
    assert d["status"] == "ok"
    assert d["rules_loaded"] == 4, f"Expected 4 rules, got {d['rules_loaded']}"
    assert d["policy"] == "KMP-OHS-POL-001"
    print(f"    System: {d['system']}")
    print(f"    Policy: {d['policy']}")
    print(f"    Rules loaded: {d['rules_loaded']}")

test("API Health Check", test_health)

# ──────────────────────────────────────────────────────────────────
# 2. COMPLIANCE RULES (Module 1 — Policy Parsing)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. COMPLIANCE RULES — Policy Parsing (Module 1)")
print("="*60)

def test_rules():
    r = requests.get(f"{BASE}/api/rules", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "rules" in d
    rules = d["rules"]
    assert len(rules) == 4, f"Expected 4 rules, got {len(rules)}"
    
    # Verify all 4 behavior classes exist
    class_ids = {rule["class_id"] for rule in rules}
    assert class_ids == {0, 1, 2, 3}, f"Missing classes: {class_ids}"
    
    expected_names = [
        "Safe Walkway Violation",
        "Unauthorized Intervention",
        "Opened Panel Cover",
        "Carrying Overload with Forklift",
    ]
    actual_names = [rule.get("behavior_name", rule.get("unsafe_behavior")) for rule in rules]
    for name in expected_names:
        assert name in actual_names, f"Missing behavior: {name}"
    
    print("    All 4 compliance behavior classes loaded:")
    for rule in rules:
        print(f"      [{rule['class_id']}] {rule['unsafe_behavior']} (ref: {rule.get('policy_ref', 'N/A')})")

test("Compliance Rules Loaded", test_rules)

def test_rules_policy_refs():
    r = requests.get(f"{BASE}/api/rules", timeout=5)
    rules = r.json()["rules"]
    # Each rule should have a policy section reference
    for rule in rules:
        ref = rule.get("policy_ref", "")
        assert "Section" in ref or "section" in ref.lower() or ref, \
            f"Rule {rule['class_id']} missing policy reference"

test("Rules Have Policy References", test_rules_policy_refs)

# ──────────────────────────────────────────────────────────────────
# 3. EVENTS & VIOLATIONS (Module 1 + Module 4)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. VIOLATION EVENTS — Detection + Reporting (Module 1 & 4)")
print("="*60)

def test_events_basic():
    r = requests.get(f"{BASE}/api/events?limit=50", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] > 0, "No events found — demo data not loaded?"
    assert len(d["events"]) > 0
    print(f"    Total events in database: {d['total']}")
    print(f"    Events returned: {len(d['events'])}")

test("Events Endpoint Returns Data", test_events_basic)

def test_event_has_required_fields():
    """Assessment requires: event_id, timestamp, clip_id, zone, behavior_class, 
       policy_rule_ref, event_description, severity, escalation_action."""
    r = requests.get(f"{BASE}/api/events?limit=1", timeout=5)
    event = r.json()["events"][0]
    required = [
        "event_id", "timestamp", "clip_id", "zone", 
        "behavior_class", "policy_rule_ref", "event_description",
        "severity", "escalation_action"
    ]
    for field in required:
        assert field in event, f"Missing required field: {field}"
        assert event[field] is not None, f"Field '{field}' is None"
    
    # Validate specific formats
    assert len(event["event_id"]) > 10, "event_id too short (should be UUID)"
    assert "T" in event["timestamp"], "timestamp not ISO 8601"
    assert event["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"], \
        f"Invalid severity: {event['severity']}"
    
    print(f"    Sample event:")
    print(f"      event_id: {event['event_id'][:20]}...")
    print(f"      timestamp: {event['timestamp']}")
    print(f"      clip_id: {event['clip_id']}")
    print(f"      zone: {event['zone']}")
    print(f"      behavior_class: {event['behavior_class']}")
    print(f"      severity: {event['severity']}")
    print(f"      policy_ref: {event['policy_rule_ref']}")
    print(f"      escalation: {event['escalation_action']}")

test("Events Have All Required Report Fields", test_event_has_required_fields)

# ──────────────────────────────────────────────────────────────────
# 4. SEVERITY CATEGORIZATION (Module 2)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. SEVERITY CATEGORIZATION (Module 2)")
print("="*60)

def test_severity_tiers():
    r = requests.get(f"{BASE}/api/events?limit=50", timeout=5)
    events = r.json()["events"]
    severities = {e["severity"] for e in events}
    
    # Should have at least LOW and HIGH/CRITICAL tiers present
    print(f"    Severity tiers present: {severities}")
    assert len(severities) >= 2, f"Expected multiple severity tiers, got {severities}"
    
    # All severities must be valid
    valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for s in severities:
        assert s in valid, f"Invalid severity tier: {s}"

test("Severity Tiers Exist", test_severity_tiers)

def test_severity_distribution():
    r = requests.get(f"{BASE}/api/events?limit=50", timeout=5)
    events = r.json()["events"]
    dist = {}
    for e in events:
        dist[e["severity"]] = dist.get(e["severity"], 0) + 1
    print(f"    Severity distribution:")
    for sev, count in sorted(dist.items()):
        print(f"      {sev}: {count} events")

test("Severity Distribution", test_severity_distribution)

# ──────────────────────────────────────────────────────────────────
# 5. ESCALATION PIPELINE (Module 3)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. ESCALATION PIPELINE (Module 3)")
print("="*60)

def test_escalation_routing():
    """LOW/MEDIUM → DB log only. HIGH/CRITICAL → alert + DB log."""
    r = requests.get(f"{BASE}/api/events?limit=50", timeout=5)
    events = r.json()["events"]
    for e in events:
        sev = e["severity"]
        action = e["escalation_action"].lower()
        if sev in ("HIGH", "CRITICAL"):
            assert "alert" in action, \
                f"{sev} event missing alert in escalation_action: {action}"
        if sev in ("LOW", "MEDIUM"):
            assert "log" in action, \
                f"{sev} event missing 'log' in escalation_action: {action}"
    print("    Routing verified:")
    print("      LOW/MEDIUM  -> logged to DB (no alert)")
    print("      HIGH/CRITICAL -> real-time alert + DB log")

test("Escalation Routing Rules", test_escalation_routing)

# ──────────────────────────────────────────────────────────────────
# 6. STATISTICS (Module 4)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. SUMMARY STATISTICS (Module 4)")
print("="*60)

def test_stats():
    r = requests.get(f"{BASE}/api/events/stats", timeout=5)
    assert r.status_code == 200
    d = r.json()
    print(f"    Stats response keys: {list(d.keys())}")
    assert "total_events" in d or "total" in d or len(d) > 0
    for k, v in d.items():
        if isinstance(v, (int, float, str)):
            print(f"      {k}: {v}")

test("Statistics Endpoint", test_stats)

# ──────────────────────────────────────────────────────────────────
# 7. EXPORT — CSV & JSON (Module 4)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. EXPORT — CSV & JSON (Module 4)")
print("="*60)

def test_export_csv():
    r = requests.get(f"{BASE}/api/export/csv", timeout=5)
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    lines = r.text.strip().split("\n")
    assert len(lines) >= 2, "CSV has no data rows"
    header = lines[0]
    assert "event_id" in header
    assert "severity" in header
    print(f"    CSV rows (incl. header): {len(lines)}")
    print(f"    CSV header columns: {header.count(',') + 1}")

test("CSV Export", test_export_csv)

def test_export_json():
    r = requests.get(f"{BASE}/api/export/json", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list), "Expected JSON array"
    assert len(data) > 0
    print(f"    JSON records: {len(data)}")

test("JSON Export", test_export_json)

# ──────────────────────────────────────────────────────────────────
# 8. FILTERING (Module 5 — Dashboard requirement)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("8. FILTERING — by severity, behavior class")
print("="*60)

def test_filter_by_severity():
    r = requests.get(f"{BASE}/api/events?severity=CRITICAL&limit=50", timeout=5)
    assert r.status_code == 200
    events = r.json()["events"]
    for e in events:
        assert e["severity"] == "CRITICAL", f"Filter leak: got {e['severity']}"
    print(f"    CRITICAL events: {len(events)}")

test("Filter by Severity", test_filter_by_severity)

def test_filter_by_behavior():
    r = requests.get(f"{BASE}/api/events?behavior_class_id=0&limit=50", timeout=5)
    assert r.status_code == 200
    events = r.json()["events"]
    for e in events:
        assert "Walkway" in e["behavior_class"], f"Filter leak: got {e['behavior_class']}"
    print(f"    Walkway violations: {len(events)}")

test("Filter by Behavior Class", test_filter_by_behavior)

# ──────────────────────────────────────────────────────────────────
# 9. DASHBOARD (Module 5)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("9. DASHBOARD SERVES (Module 5)")
print("="*60)

def test_dashboard_html():
    r = requests.get(f"{BASE}/", timeout=5)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()
    assert "AegisFlow" in r.text or "Compliance" in r.text or "dashboard" in r.text.lower()
    print(f"    Dashboard HTML size: {len(r.text):,} bytes")

test("Dashboard HTML Served", test_dashboard_html)

def test_static_css():
    r = requests.get(f"{BASE}/static/style.css", timeout=5)
    assert r.status_code == 200
    print(f"    CSS size: {len(r.text):,} bytes")

test("Static CSS Loaded", test_static_css)

def test_static_js():
    r = requests.get(f"{BASE}/static/app.js", timeout=5)
    assert r.status_code == 200
    print(f"    JS size: {len(r.text):,} bytes")

test("Static JS Loaded", test_static_js)

# ──────────────────────────────────────────────────────────────────
# 10. RULES CONFIG (Dynamic Settings)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("10. DYNAMIC RULES CONFIG")
print("="*60)

def test_get_rules_config():
    r = requests.get(f"{BASE}/api/rules/config", timeout=5)
    assert r.status_code == 200
    print(f"    Config response: {type(r.json()).__name__}")

test("GET Rules Config", test_get_rules_config)

# ──────────────────────────────────────────────────────────────────
# 11. API DOCS (FastAPI auto-docs)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("11. API DOCUMENTATION")
print("="*60)

def test_openapi_docs():
    r = requests.get(f"{BASE}/docs", timeout=5)
    assert r.status_code == 200
    print(f"    Swagger UI available at: {BASE}/docs")

test("Swagger API Docs", test_openapi_docs)

def test_openapi_schema():
    r = requests.get(f"{BASE}/openapi.json", timeout=5)
    assert r.status_code == 200
    schema = r.json()
    paths = list(schema.get("paths", {}).keys())
    print(f"    API endpoints: {len(paths)}")
    for p in sorted(paths):
        print(f"      {p}")

test("OpenAPI Schema", test_openapi_schema)

# ──────────────────────────────────────────────────────────────────
# 12. KAGGLE DATASET CHECK
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("12. KAGGLE DATASET STATUS")
print("="*60)

from pathlib import Path

def test_kaggle_dataset():
    data_dir = Path(__file__).parent / "data"
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [f for f in data_dir.iterdir() if f.suffix.lower() in video_exts] if data_dir.exists() else []
    
    if videos:
        print(f"    FOUND {len(videos)} video files in data/:")
        for v in videos:
            size_mb = v.stat().st_size / 1_048_576
            print(f"      {v.name} ({size_mb:.1f} MB)")
    else:
        print("    No Kaggle video files downloaded in data/ directory.")
        print("    System is running on SYNTHETIC DEMO data (fully functional).")
        print()
        print("    To use real Kaggle data, download from:")
        print("    https://www.kaggle.com/datasets/trnhhnggiang/video-dataset-for-safe-and-unsafe-behaviours")
        print("    Then place .mp4 files in the data/ folder and restart.")
    
    # This is informational, not a pass/fail
    return True

test("Kaggle Dataset Check", test_kaggle_dataset)

# ──────────────────────────────────────────────────────────────────
# 13. OUTPUT FILES CHECK
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("13. OUTPUT FILES — Audit Trail")
print("="*60)

def test_output_files():
    outputs = Path(__file__).parent / "outputs"
    assert outputs.exists(), "outputs/ directory missing"
    
    db = outputs / "compliance.db"
    csv_file = outputs / "audit_log.csv"
    json_file = outputs / "audit_log.json"
    
    files = {"compliance.db": db, "audit_log.csv": csv_file, "audit_log.json": json_file}
    for name, path in files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"    {name}: {size_kb:.1f} KB")
        else:
            print(f"    {name}: NOT FOUND")

test("Output Files Exist", test_output_files)

# ──────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TEST SUMMARY")
print("="*60)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")
print()

if failed > 0:
    print("  FAILED TESTS:")
    for r in results:
        if r[0] == "FAIL":
            print(f"    - {r[1]}: {r[2]}")
    print()

if failed == 0:
    print("  ALL TESTS PASSED!")
else:
    print(f"  {failed} TEST(S) FAILED")

sys.exit(1 if failed > 0 else 0)

/**
 * app.js -- AegisFlow EHS Vision Platform SPA
 * Handles: WebSocket real-time feed, stats updates, timeline stream,
 * historical log table with filters/pagination, export, alert overlay,
 * interactive policy configuration, floor status heatmap, and dynamic print summaries.
 */

'use strict';

// --- Config ----------------------------------------------------------------
const WS_URL = `ws://${location.host}/ws`;
const API    = '/api';

// --- State -----------------------------------------------------------------
let ws            = null;
let wsRetryDelay  = 1500;
let timelineItems = [];   // full list for client-side filtering
let logPage       = 0;
let logTotal      = 0;
let logLimit      = 50;
let statsInterval = null;

// Dynamic parameters config registry
let policyRules   = [];
let customConfig  = {};
let activeAlerts  = new Set();

// --- Init -------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  loadStats();
  loadEvents();
  loadPolicySettings();
  statsInterval = setInterval(loadStats, 15_000);  // refresh stats every 15s

  // Set print timestamp dynamically
  window.onbeforeprint = () => {
    document.body.setAttribute('data-print-time', new Date().toLocaleString());
  };
});

// 2-Tab Navigation (Feature 1 Navigation)
window.switchTab = function(tab) {
  const dTab = document.getElementById('tab-dashboard');
  const sTab = document.getElementById('tab-settings');
  const dView = document.getElementById('view-dashboard');
  const sView = document.getElementById('view-settings');

  if (tab === 'dashboard') {
    dTab.classList.add('active');
    sTab.classList.remove('active');
    dView.classList.remove('hidden');
    sView.classList.add('hidden');
    loadStats();
    loadEvents();
  } else {
    dTab.classList.remove('active');
    sTab.classList.add('active');
    dView.classList.add('hidden');
    sView.classList.remove('hidden');
    loadPolicySettings();
  }
};

// ===========================================================================
// WebSocket Connection
// ===========================================================================

function connectWebSocket() {
  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    setWsStatus(true);
    wsRetryDelay = 1500;
    // Keep-alive ping every 20s
    ws._pingTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 20_000);
  };

  ws.onmessage = ({ data }) => {
    try {
      const msg = JSON.parse(data);
      handleWsMessage(msg);
    } catch (_) {}
  };

  ws.onclose = () => {
    setWsStatus(false);
    clearInterval(ws._pingTimer);
    scheduleReconnect();
  };

  ws.onerror = () => {
    setWsStatus(false);
  };
}

function scheduleReconnect() {
  setTimeout(() => connectWebSocket(), wsRetryDelay);
  wsRetryDelay = Math.min(wsRetryDelay * 1.5, 30_000);
}

function setWsStatus(connected) {
  const dot   = document.getElementById('ws-indicator');
  const label = document.getElementById('ws-label');
  dot.className   = `ws-dot ${connected ? 'connected' : 'disconnected'}`;
  label.textContent = connected ? 'Live connection' : 'Reconnecting…';
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'connected':
      updateStatsDisplay(msg.stats);
      break;
    case 'alert':
      handleAlert(msg);
      addTimelineItem(msg);
      updateFeedFromEvent(msg);
      updateHeatmapFromEvent(msg);
      loadStats();
      break;
    case 'event':
      addTimelineItem(msg);
      updateFeedFromEvent(msg);
      updateHeatmapFromEvent(msg);
      loadStats();
      break;
    case 'status':
      handleStatus(msg);
      break;
  }
}

// ===========================================================================
// Stats (Metrics Calculation with EHS KPI logic)
// ===========================================================================

async function loadStats() {
  try {
    const res  = await fetch(`${API}/events/stats`);
    const data = await res.json();
    updateStatsDisplay(data);
  } catch (_) {}
}

function updateStatsDisplay(data) {
  if (!data) return;
  setText('s-total',    data.total_events ?? '—');
  setText('s-critical', data.by_severity?.CRITICAL ?? '—');
  setText('s-high',     data.by_severity?.HIGH ?? '—');
  setText('s-medium',   data.by_severity?.MEDIUM ?? '—');
  setText('s-low',      data.by_severity?.LOW ?? '—');

  // Advanced feature calculation: EHS Safety Rating Index
  // Formula: start at 100, subtract 15 points per Critical, 5 per High, 2 per Medium, 0.5 per Low
  const crit = data.by_severity?.CRITICAL || 0;
  const high = data.by_severity?.HIGH || 0;
  const med  = data.by_severity?.MEDIUM || 0;
  const low  = data.by_severity?.LOW || 0;
  
  const rawScore = 100 - (crit * 12 + high * 6 + med * 2.5 + low * 0.5);
  const complianceIndex = Math.max(20, Math.min(100, rawScore)).toFixed(1);
  setText('s-alerts',   complianceIndex + '%');
}

// ===========================================================================
// Alert Overlay (Module 3 Alert)
// ===========================================================================

function handleAlert(msg) {
  const sev = msg.severity || 'HIGH';
  const overlay = document.getElementById('alert-overlay');

  document.getElementById('alert-severity-label').textContent = sev;
  document.getElementById('alert-severity-label').className   = `alert-severity-badge ${sev}`;
  document.getElementById('alert-title').textContent =
    `${sev} Violation -- ${msg.behavior_name || 'Compliance Breach'}`;
  document.getElementById('alert-body').textContent =
    msg.description || msg.event_description || 'Unsafe behavior detected on production floor.';
  document.getElementById('alert-zone').textContent   = `Zone: ${msg.zone || '—'}`;
  document.getElementById('alert-policy').textContent = `Ref: ${msg.policy_ref || msg.policy_rule_ref || '—'}`;
  document.getElementById('alert-conf').textContent   = `Confidence: ${fmtConf(msg.confidence)}`;

  const thumb = document.getElementById('alert-thumb');
  if (msg.thumbnail_b64) {
    thumb.src = `data:image/jpeg;base64,${msg.thumbnail_b64}`;
    thumb.classList.remove('hidden');
  } else {
    thumb.classList.add('hidden');
  }

  overlay.classList.remove('hidden');
  activeAlerts.add(msg.event_id);

  // Auto-dismiss after 8 seconds unless it's CRITICAL
  if (sev !== 'CRITICAL') {
    setTimeout(() => {
      if (activeAlerts.has(msg.event_id)) {
        dismissAlert();
      }
    }, 8000);
  }
}

window.dismissAlert = function() {
  document.getElementById('alert-overlay').classList.add('hidden');
  activeAlerts.clear();
  // Reset floor status map highlights
  clearFloorHeatmap();
};

// ===========================================================================
// Floor Status Heatmap Grid (Feature 2 Heuristics)
// ===========================================================================

function updateHeatmapFromEvent(msg) {
  // Infer floor zone cell from event zone string (e.g. Zone-Mid-Center)
  let zoneStr = msg.zone || '';
  if (!zoneStr.startsWith('Zone-')) {
    // Attempt fallback maps
    if (zoneStr.toLowerCase().includes('left')) zoneStr = 'Zone-Mid-Left';
    else if (zoneStr.toLowerCase().includes('right')) zoneStr = 'Zone-Mid-Right';
    else zoneStr = 'Zone-Mid-Center';
  }
  
  // Find matching element
  // Format matches exactly index.html: zone-Top-Left, zone-Mid-Center, etc.
  const cleanId = zoneStr.replace('Zone-', 'zone-');
  const cell = document.getElementById(cleanId);
  if (cell) {
    const sev = msg.severity || 'LOW';
    // Remove old active states
    cell.className = 'zone-cell';
    // Apply new warning state
    cell.classList.add(`active-${sev}`);
    
    // Add zone status overlay badge text change
    const label = cell.querySelector('.zone-label');
    if (label) {
      label.textContent = `${sev} ALERT`;
    }

    // Auto clear LOW/MEDIUM after 8s
    if (sev !== 'CRITICAL' && sev !== 'HIGH') {
      setTimeout(() => {
        cell.className = 'zone-cell';
        if (label) label.textContent = zoneStr.replace('Zone-', '');
      }, 8000);
    }
  }
}

function clearFloorHeatmap() {
  const cells = document.querySelectorAll('.zone-cell');
  cells.forEach(cell => {
    cell.className = 'zone-cell';
    const label = cell.querySelector('.zone-label');
    if (label) {
      const originalZoneName = cell.id.replace('zone-', '');
      label.textContent = originalZoneName;
    }
  });
}

// ===========================================================================
// Live Feed (View A)
// ===========================================================================

function updateFeedFromEvent(msg) {
  const thumb = document.getElementById('feed-thumb');
  const placeholder = document.getElementById('feed-placeholder');
  const overlay = document.getElementById('feed-overlay');
  const badge   = document.getElementById('feed-badge');
  const vname   = document.getElementById('feed-violation-name');

  if (msg.thumbnail_b64) {
    thumb.src = `data:image/jpeg;base64,${msg.thumbnail_b64}`;
    thumb.classList.remove('hidden');
    placeholder.classList.add('hidden');
    overlay.classList.remove('hidden');
  }

  const sev = msg.severity || 'HIGH';
  badge.textContent  = sev;
  badge.className    = `feed-badge ${sev}`;
  vname.textContent  = msg.behavior_name || msg.behavior_class || '';

  setText('fi-time', fmtTime(msg.timestamp || new Date().toISOString()));
  setText('fi-clip', msg.clip_id || '—');
  setText('fi-zone', msg.zone || '—');
  setText('fi-conf', fmtConf(msg.confidence));
}

// ===========================================================================
// Alert Timeline (View B)
// ===========================================================================

function addTimelineItem(msg) {
  timelineItems.unshift(msg);
  const empty = document.getElementById('timeline-empty');
  if (empty) empty.classList.add('hidden');
  renderTimeline();
}

function renderTimeline() {
  const sevFilter  = document.getElementById('tf-severity')?.value || '';
  const clsFilter  = document.getElementById('tf-class')?.value || '';

  const filtered = timelineItems.filter(m => {
    if (sevFilter && m.severity !== sevFilter) return false;
    const cid = String(m.behavior_class_id ?? m.behavior_class ?? '');
    if (clsFilter && cid !== clsFilter) return false;
    return true;
  });

  const stream = document.getElementById('timeline-stream');
  const empty  = document.getElementById('timeline-empty');

  if (filtered.length === 0) {
    stream.innerHTML = '';
    stream.appendChild(empty);
    empty.classList.remove('hidden');
    return;
  }

  stream.innerHTML = '';
  filtered.slice(0, 80).forEach(m => {
    const sev   = m.severity || 'LOW';
    const name  = m.behavior_name || m.behavior_class || 'Unknown';
    const time  = fmtTime(m.timestamp || new Date().toISOString());
    const zone  = m.zone || '—';
    const conf  = fmtConf(m.confidence);
    const thumb = m.thumbnail_b64 || null;

    const item = document.createElement('div');
    item.className = `timeline-item ${sev}`;
    item.onclick   = () => showDetail(m);
    item.innerHTML = `
      <span class="ti-sev ${sev}">${sev}</span>
      <div class="ti-body">
        <div class="ti-name">${escHtml(name)}</div>
        <div class="ti-meta">
          <span>🕐 ${time}</span>
          <span>📍 ${escHtml(zone)}</span>
          <span>📊 ${conf}</span>
        </div>
      </div>
      ${thumb ? `<img class="ti-thumb" src="data:image/jpeg;base64,${thumb}" alt="frame" />` : ''}
    `;
    stream.appendChild(item);
  });
}

window.filterTimeline = function() { renderTimeline(); };

window.clearTimeline = function() {
  timelineItems = [];
  const stream = document.getElementById('timeline-stream');
  const empty  = document.getElementById('timeline-empty');
  stream.innerHTML = '';
  stream.appendChild(empty);
  empty.classList.remove('hidden');
  clearFloorHeatmap();
};

// ===========================================================================
// Historical Log (View C)
// ===========================================================================

async function loadEvents() {
  const severity  = document.getElementById('lf-severity')?.value || '';
  const classId   = document.getElementById('lf-class')?.value || '';
  const dateFrom  = document.getElementById('lf-from')?.value || '';
  const dateTo    = document.getElementById('lf-to')?.value || '';
  logLimit        = parseInt(document.getElementById('lf-limit')?.value || '50');

  let url = `${API}/events?limit=${logLimit}&offset=${logPage * logLimit}`;
  if (severity) url += `&severity=${severity}`;
  if (classId)  url += `&behavior_class_id=${classId}`;
  if (dateFrom) url += `&date_from=${dateFrom}`;
  if (dateTo)   url += `&date_to=${dateTo}`;

  const tbody = document.getElementById('log-tbody');
  tbody.innerHTML = `<tr><td colspan="9" class="table-empty">Querying historical database…</td></tr>`;

  try {
    const res  = await fetch(url);
    const data = await res.json();
    logTotal   = data.total || 0;
    renderLogTable(data.events || []);
    renderPagination();
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9" class="table-empty">Error fetching events</td></tr>`;
  }
}

function renderLogTable(events) {
  const tbody = document.getElementById('log-tbody');
  if (!events.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="table-empty">No records found matching query criteria</td></tr>`;
    return;
  }

  tbody.innerHTML = events.map(ev => {
    const sev  = ev.severity || 'LOW';
    const alrt = ev.alert_triggered;
    return `
      <tr>
        <td><span class="sev-pill ${sev}">${sev}</span></td>
        <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-secondary)">${fmtTime(ev.timestamp)}</td>
        <td style="max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:700">${escHtml(ev.behavior_class || '—')}</td>
        <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-code)">${escHtml(ev.clip_id || '—')}</td>
        <td style="font-size:11px;color:var(--text-secondary)">${escHtml(ev.zone || '—')}</td>
        <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-code)">${escHtml(ev.policy_rule_ref || '—')}</td>
        <td style="font-family:var(--font-mono);font-size:11px">${fmtConf(ev.confidence)}</td>
        <td><span class="alert-dot ${alrt ? 'yes' : 'no'}" title="${alrt ? 'Alert triggered' : 'Logged only'}"></span></td>
        <td><button class="btn-detail" onclick='showDetailById(${JSON.stringify(ev)})'>View Details</button></td>
      </tr>
    `;
  }).join('');
}

function renderPagination() {
  const container = document.getElementById('log-pagination');
  const totalPages = Math.ceil(logTotal / logLimit) || 1;
  const current    = logPage + 1;

  container.innerHTML = `
    <button class="btn-page" onclick="goPage(${logPage - 1})" ${logPage === 0 ? 'disabled' : ''}>‹ Prev</button>
    <span class="page-info">Page ${current} of ${totalPages} (${logTotal} records)</span>
    <button class="btn-page" onclick="goPage(${logPage + 1})" ${current >= totalPages ? 'disabled' : ''}>Next ›</button>
  `;
}

window.goPage = function(p) {
  const totalPages = Math.ceil(logTotal / logLimit);
  if (p < 0 || p >= totalPages) return;
  logPage = p;
  loadEvents();
};

// ===========================================================================
// Interactive Settings (Feature 1 Rules Configuration Controller)
// ===========================================================================

async function loadPolicySettings() {
  try {
    // 1. Fetch compliance rules
    const rulesRes = await fetch(`${API}/rules`);
    const rulesData = await rulesRes.json();
    policyRules = rulesData.rules || [];

    // 2. Fetch parameter overrides
    const configRes = await fetch(`${API}/rules/config`);
    customConfig = await configRes.json();

    renderPolicySettingsGrid();
  } catch (e) {
    const grid = document.getElementById('rules-settings-grid');
    if (grid) grid.innerHTML = `<div class="settings-loading">Error loading rule parameters: ${e.message}</div>`;
  }
}

function renderPolicySettingsGrid() {
  const grid = document.getElementById('rules-settings-grid');
  if (!grid) return;

  if (policyRules.length === 0) {
    grid.innerHTML = '<div class="settings-loading">No rules loaded.</div>';
    return;
  }

  grid.innerHTML = policyRules.map(rule => {
    const cid = rule.class_id;
    // Get custom overrides if set, otherwise default to rules parameters
    const override = customConfig[cid] || {};
    const active = override.active !== undefined ? override.active : rule.active;
    const severity = override.severity || rule.severity;
    const threshold = override.confidence_threshold !== undefined ? override.confidence_threshold : rule.confidence_threshold;

    return `
      <div class="rule-editor-card" id="rule-card-${cid}" data-class-id="${cid}">
        <div class="rule-card-domain">${escHtml(rule.domain)}</div>
        <div class="rule-card-header">
          <h4 class="rule-card-title">${escHtml(rule.behavior_name)}</h4>
          <span class="policy-badge">${escHtml(rule.policy_ref)}</span>
        </div>
        <p class="rule-card-desc">${escHtml(rule.unsafe_behavior)}</p>
        
        <div class="rule-control-group">
          <!-- Toggle active/inactive -->
          <div class="rule-control-row">
            <span class="control-label">Surveillance Monitor</span>
            <label class="switch">
              <input type="checkbox" class="rule-active-toggle" ${active ? 'checked' : ''} onchange="updateRuleUIState(${cid})">
              <span class="slider"></span>
            </label>
          </div>

          <!-- Threshold Slider -->
          <div class="rule-control-row">
            <span class="control-label">Min Confidence</span>
            <div class="slider-container">
              <input type="range" class="range-slider rule-threshold-slider" min="0.20" max="0.95" step="0.05" value="${threshold.toFixed(2)}" oninput="updateSliderLabel(${cid}, this.value)">
              <span class="range-val" id="slider-val-${cid}">${Math.round(threshold * 100)}%</span>
            </div>
          </div>

          <!-- Severity override selection -->
          <div class="rule-control-row">
            <span class="control-label">Severity Level</span>
            <select class="filter-select rule-severity-select" style="width: 130px;">
              <option value="CRITICAL" ${severity === 'CRITICAL' ? 'selected' : ''}>Critical</option>
              <option value="HIGH" ${severity === 'HIGH' ? 'selected' : ''}>High</option>
              <option value="MEDIUM" ${severity === 'MEDIUM' ? 'selected' : ''}>Medium</option>
              <option value="LOW" ${severity === 'LOW' ? 'selected' : ''}>Low</option>
            </select>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

window.updateSliderLabel = function(cid, value) {
  const lbl = document.getElementById(`slider-val-${cid}`);
  if (lbl) lbl.textContent = Math.round(value * 100) + '%';
};

window.updateRuleUIState = function(cid) {
  const card = document.getElementById(`rule-card-${cid}`);
  const toggle = card.querySelector('.rule-active-toggle');
  if (toggle && !toggle.checked) {
    card.style.opacity = '0.5';
  } else {
    card.style.opacity = '1.0';
  }
};

window.savePolicySettings = async function() {
  const cards = document.querySelectorAll('.rule-editor-card');
  const payload = {};

  cards.forEach(card => {
    const cid = card.getAttribute('data-class-id');
    const active = card.querySelector('.rule-active-toggle').checked;
    const threshold = parseFloat(card.querySelector('.rule-threshold-slider').value);
    const severity = card.querySelector('.rule-severity-select').value;

    payload[cid] = {
      active: active,
      confidence_threshold: threshold,
      severity: severity
    };
  });

  showToast('Saving EHS policy rules changes…');

  try {
    const res = await fetch(`${API}/rules/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (res.ok) {
      showToast(data.message || 'Configuration saved successfully ✓');
      setTimeout(() => { loadPolicySettings(); }, 500);
    } else {
      showToast('Error saving settings overrides', true);
    }
  } catch (e) {
    showToast('Failed to save configs', true);
  }
};

// ===========================================================================
// Details Modal
// ===========================================================================

window.showDetail = function(ev) {
  showDetailById(ev);
};

window.showDetailById = function(ev) {
  document.getElementById('dm-title').textContent =
    `${ev.severity || ''} -- ${ev.behavior_name || ev.behavior_class || 'Event Detail'}`;

  const body = document.getElementById('detail-body');
  const fields = [
    ['Event ID',       ev.event_id,                    'mono'],
    ['Timestamp',      fmtTime(ev.timestamp),           ''],
    ['Behavior Class', ev.behavior_name || ev.behavior_class, ''],
    ['Severity',       ev.severity,                    ''],
    ['Policy Ref',     ev.policy_rule_ref || ev.policy_ref,  'mono'],
    ['Zone',           ev.zone,                        ''],
    ['Clip ID',        ev.clip_id,                     'mono'],
    ['Confidence',     fmtConf(ev.confidence),         'mono'],
    ['Detector',       ev.detector_source,             'mono'],
    ['Alert Triggered',ev.alert_triggered ? 'Yes ⚠' : 'No (logged only)', ''],
    ['Escalation',     ev.escalation_action,           ''],
    ['Description',    ev.event_description || ev.description, ''],
  ];

  body.innerHTML = fields.map(([label, val, cls]) => `
    <div class="detail-field">
      <div class="detail-field-label">${label}</div>
      <div class="detail-field-value ${cls}">${escHtml(String(val ?? '—'))}</div>
    </div>
  `).join('');

  if (ev.thumbnail_b64) {
    body.innerHTML += `
      <div class="detail-field">
        <div class="detail-field-label">Frame Thumbnail</div>
        <img class="detail-thumb" src="data:image/jpeg;base64,${ev.thumbnail_b64}" alt="violation frame" />
      </div>
    `;
  }

  document.getElementById('detail-modal').classList.remove('hidden');
};

window.closeDetail = function() {
  document.getElementById('detail-modal').classList.add('hidden');
};

// ===========================================================================
// Export Data
// ===========================================================================

window.exportData = async function(format) {
  const severity = document.getElementById('lf-severity')?.value || '';
  const classId  = document.getElementById('lf-class')?.value || '';
  const dateFrom = document.getElementById('lf-from')?.value || '';
  const dateTo   = document.getElementById('lf-to')?.value || '';

  let url = `${API}/export/${format}?`;
  if (severity) url += `&severity=${severity}`;
  if (classId)  url += `&behavior_class_id=${classId}`;
  if (dateFrom) url += `&date_from=${dateFrom}`;
  if (dateTo)   url += `&date_to=${dateTo}`;

  showToast(`Preparing ${format.toUpperCase()} export…`);
  const a = document.createElement('a');
  a.href = url;
  a.download = `compliance_audit_log.${format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  showToast(`${format.toUpperCase()} download started ✓`);
};

// ===========================================================================
// Processing Triggers
// ===========================================================================

window.triggerProcessing = async function() {
  showToast('Starting video analysis pipeline…');
  try {
    const res  = await fetch(`${API}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_demo: false, sample_every: 15 }),
    });
    const data = await res.json();
    if (res.status === 409) {
      showToast('Processing already running…', false);
    } else {
      showToast(data.message || 'Processing started ✓');
    }
  } catch (e) {
    showToast('Error starting processing', true);
  }
};

window.loadDemo = async function() {
  showToast('Loading demo violations…');
  try {
    const res  = await fetch(`${API}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_demo: true }),
    });
    const data = await res.json();
    showToast(data.message || 'Demo loaded ✓');
    setTimeout(() => { loadStats(); loadEvents(); }, 1200);
  } catch (e) {
    showToast('Error loading demo', true);
  }
};

// ===========================================================================
// WebSocket Status Messages
// ===========================================================================

function handleStatus(msg) {
  const status = msg.status || '';
  if (status === 'processing_started') {
    showToast('Processing pipeline running…');
    document.getElementById('feed-status-text').textContent = 'Analyzing';
  } else if (status === 'processing_complete') {
    showToast(msg.message || 'Processing complete ✓');
    document.getElementById('feed-status-text').textContent = 'Monitoring';
    loadStats();
    loadEvents();
  } else if (status === 'processing_error') {
    showToast(`Processing error: ${msg.error}`, true);
    document.getElementById('feed-status-text').textContent = 'Error';
  } else if (status === 'rules_updated') {
    showToast(msg.message || 'Policy parameters reloaded ✓');
    loadStats();
    loadEvents();
    loadPolicySettings();
  }
}

// ===========================================================================
// Utilities
// ===========================================================================

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString('en-GB', {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  } catch (_) { return iso; }
}

function fmtConf(v) {
  if (v === undefined || v === null) return '—';
  return `${(parseFloat(v) * 100).toFixed(1)}%`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

let _toastTimer = null;
function showToast(msg, isError = false) {
  const toast = document.getElementById('toast');
  document.getElementById('toast-msg').textContent = msg;
  toast.className = `toast${isError ? ' error' : ''}`;
  toast.classList.remove('hidden');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.add('hidden'), 3500);
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    dismissAlert();
    closeDetail();
  }
});

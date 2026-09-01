/* ===========================================================================
   AegisFlow EHS - Operations Dashboard
   ---------------------------------------------------------------------------
   Plain ES modules, no framework, no build step. Three views over the FastAPI
   backend, plus a policy panel that shows where each severity tier came from.

   View A  Live Feed Monitor      processed clip playback + severity overlay
   View B  Alert Timeline Stream  live WebSocket feed, strobe on HIGH/CRITICAL
   View C  Historical Log         filter by date / severity / behaviour, export
   =========================================================================== */

const API = '';
const MAX_TIMELINE = 200;

const state = {
  view: 'live',
  clips: [],
  activeClip: null,
  timeline: [],
  history: { items: [], total: 0, offset: 0, limit: 50 },
  behaviors: new Set(),
  socket: null,
  reconnectDelay: 1000,
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const titleCase = (value) =>
  String(value || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

const fmtTime = (iso) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? String(iso) : date.toISOString().slice(11, 19);
};
const fmtDateTime = (iso) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? String(iso) : date.toISOString().slice(0, 19).replace('T', ' ');
};

async function api(path, params) {
  const url = new URL(API + path, window.location.origin);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null || value === '') continue;
      if (Array.isArray(value)) value.forEach((v) => url.searchParams.append(key, v));
      else url.searchParams.set(key, value);
    }
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

/* ------------------------------------------------------------------ chrome */

function initTheme() {
  const stored = safeGet('aegisflow.theme');
  if (stored) document.documentElement.setAttribute('data-theme', stored);
  $('theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    safeSet('aegisflow.theme', next);
  });
}

// localStorage throws in some embedded contexts; a remembered theme is never worth a crash.
function safeGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function safeSet(key, value) { try { localStorage.setItem(key, value); } catch { /* ignore */ } }

function initTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  });
}

function showView(name) {
  state.view = name;
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.setAttribute('aria-selected', String(tab.dataset.view === name));
  });
  document.querySelectorAll('.view').forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
  });
  if (name === 'history') loadHistory();
  if (name === 'policy') loadPolicy();
}

function setConnection(stateName, label) {
  const node = $('conn');
  node.dataset.state = stateName;
  $('conn-label').textContent = label;
}

/* ------------------------------------------------------------------- tiles */

async function loadStats() {
  let stats;
  try {
    stats = await api('/api/stats');
  } catch {
    return;
  }
  const tiles = $('tiles');
  tiles.replaceChildren();

  const add = (label, value, meta, variant) => {
    const tile = el('div', `tile${variant ? ` tile--${variant}` : ''}`);
    tile.append(el('div', 'tile__label', label), el('div', 'tile__value', String(value)));
    if (meta) tile.append(el('div', 'tile__meta', meta));
    tiles.append(tile);
  };

  const sev = stats.by_severity || {};
  add('Total events', stats.events_recorded, `${stats.clips_processed} clips processed`, 'accent');
  add('Critical', sev.CRITICAL || 0, 'immediate danger', 'critical');
  add('High', sev.HIGH || 0, 'alert + log', 'high');
  add('Medium', sev.MEDIUM || 0, 'log only', 'medium');
  add('Low', sev.LOW || 0, 'log only', 'low');
  add('Real-time alerts', stats.alerts_total || 0, 'HIGH + CRITICAL routed');
  if (stats.vlm_calls) add('VLM consultations', stats.vlm_calls, 'ambiguous frames');
}

/* --------------------------------------------------- View A: Live Feed --- */

async function loadClips() {
  try {
    state.clips = await api('/api/clips', { limit: 200 });
  } catch {
    state.clips = [];
  }
  const list = $('clip-list');
  list.replaceChildren();

  if (!state.clips.length) {
    list.append(el('div', 'empty', 'No processed clips.'));
    return;
  }

  for (const clip of state.clips) {
    const button = el('button', 'clip');
    button.type = 'button';
    if (clip.worst_severity) {
      button.style.setProperty('--clip-accent', `var(--sev-${clip.worst_severity.toLowerCase()})`);
    } else {
      button.style.setProperty('--clip-accent', 'var(--sev-medium)');
    }
    button.append(el('span', 'clip__id', clip.clip_id));
    button.append(
      el('span', 'clip__n', clip.violation_count ? `${clip.violation_count}` : 'clear')
    );
    button.addEventListener('click', () => selectClip(clip));
    list.append(button);
  }
  if (!state.activeClip) selectClip(state.clips[0]);
}

async function selectClip(clip) {
  state.activeClip = clip;
  document.querySelectorAll('.clip').forEach((node, index) => {
    node.setAttribute('aria-current', String(state.clips[index]?.clip_id === clip.clip_id));
  });

  const stage = $('stage');
  const player = $('player');
  const tier = clip.worst_severity;

  stage.style.setProperty('--stage-accent', tier ? `var(--sev-${tier.toLowerCase()})` : 'var(--sev-medium)');
  $('stage-caption').textContent = `${clip.clip_id} — ${clip.zone || 'unassigned zone'}`;

  const status = $('stage-status');
  status.hidden = false;
  const badge = $('stage-sev');
  badge.className = `sev sev--${tier || 'none'}`;
  badge.textContent = tier || 'NO VIOLATION';
  $('stage-text').textContent = clip.violation_count
    ? `${clip.violation_count} violation(s) detected`
    : 'compliant — no violation detected';

  if (clip.has_video) {
    player.hidden = false;
    $('stage-empty').hidden = true;
    player.src = `/api/clips/${encodeURIComponent(clip.clip_id)}/video`;
    player.load();
  } else {
    player.hidden = true;
    player.removeAttribute('src');
    $('stage-empty').hidden = false;
    $('stage-empty').textContent = 'Annotated video not rendered for this clip.';
  }

  // Assignment View A: HIGH/CRITICAL must render the real-time alert visibly.
  if (tier === 'HIGH' || tier === 'CRITICAL') {
    stage.classList.remove('stage--alert');
    void stage.offsetWidth; // restart the animation
    stage.classList.add('stage--alert');
  }

  const container = $('clip-events');
  container.replaceChildren();
  try {
    const page = await api('/api/events', { clip_id: clip.clip_id, limit: 25 });
    if (!page.items.length) {
      container.append(el('div', 'empty', 'No violations recorded for this clip.'));
      return;
    }
    page.items.forEach((event) => container.append(renderEvent(event, false)));
  } catch {
    container.append(el('div', 'empty', 'Could not load events for this clip.'));
  }
}

/* --------------------------------------------------- View B: Timeline ---- */

function renderEvent(event, isNew) {
  const tier = event.severity;
  const row = el('div', `event${isNew ? ' event--new' : ''}`);
  row.style.setProperty('--event-accent', `var(--sev-${tier.toLowerCase()})`);
  if (tier === 'HIGH' || tier === 'CRITICAL') row.classList.add('event--alert');

  row.append(el('div', 'event__time', fmtTime(event.timestamp)));

  const body = el('div');
  body.append(el('div', 'event__title', titleCase(event.behavior_class)));
  body.append(el('div', 'event__desc', event.event_description));

  const meta = el('div', 'event__meta');
  meta.append(el('span', 'chip chip--policy', event.policy_rule_ref));
  meta.append(el('span', 'chip', event.zone));
  meta.append(el('span', 'chip', event.clip_id));
  meta.append(el('span', 'chip', `conf ${Number(event.confidence).toFixed(2)}`));
  meta.append(el('span', 'chip', event.detection_method));
  body.append(meta);

  if (event.severity_rationale) {
    const rationale = el('div', 'rationale');
    rationale.append(el('strong', null, 'Why this tier: '));
    rationale.append(document.createTextNode(event.severity_rationale));
    body.append(rationale);
  }
  row.append(body);

  const side = el('div', 'event__side');
  side.append(el('span', `sev sev--${tier}`, tier));
  side.append(el('span', 'chip', event.escalation_action));
  row.append(side);

  return row;
}

function pushTimelineEvent(event) {
  state.timeline.unshift(event);
  if (state.timeline.length > MAX_TIMELINE) state.timeline.length = MAX_TIMELINE;

  const container = $('timeline');
  $('timeline-empty')?.remove();
  container.prepend(renderEvent(event, true));
  while (container.children.length > MAX_TIMELINE) container.lastElementChild.remove();
  $('timeline-count').textContent = String(state.timeline.length);

  if (event.severity === 'HIGH' || event.severity === 'CRITICAL') showAlertBanner(event);
}

function showAlertBanner(event) {
  document.querySelector('.alert-banner')?.remove();
  const banner = el('div', 'alert-banner');
  banner.setAttribute('role', 'alert');
  banner.style.setProperty('--banner-accent', `var(--sev-${event.severity.toLowerCase()})`);

  const body = el('div', 'alert-banner__body');
  body.append(el('div', 'alert-banner__title', `${event.severity} — ${titleCase(event.behavior_class)}`));
  body.append(el('div', 'alert-banner__text', `${event.zone} · ${event.clip_id} · ${event.policy_rule_ref}`));
  banner.append(el('span', `sev sev--${event.severity}`, event.severity), body);

  const close = el('button', 'alert-banner__close', '×');
  close.setAttribute('aria-label', 'Dismiss alert');
  close.addEventListener('click', () => banner.remove());
  banner.append(close);

  document.body.append(banner);
  setTimeout(() => banner.remove(), 9000);
}

/* ---------------------------------------------------- View C: History ---- */

function filterValues() {
  const form = $('filters');
  const data = new FormData(form);
  const multi = (name) =>
    Array.from(form.elements[name].selectedOptions || []).map((option) => option.value);
  return {
    date_from: data.get('date_from') ? `${data.get('date_from')}T00:00:00` : '',
    date_to: data.get('date_to') ? `${data.get('date_to')}T23:59:59` : '',
    severity: multi('severity'),
    behavior_class: multi('behavior_class'),
    limit: Number(data.get('limit') || 50),
  };
}

async function loadHistory(offset = 0) {
  const filters = filterValues();
  state.history.limit = filters.limit;
  state.history.offset = offset;

  let page;
  try {
    page = await api('/api/events', { ...filters, offset });
  } catch (error) {
    $('history-summary').textContent = `error: ${error.message}`;
    return;
  }

  state.history.items = page.items;
  state.history.total = page.total;

  const body = $('history-body');
  body.replaceChildren();

  if (!page.items.length) {
    const row = el('tr');
    const cell = el('td', 'empty', 'No records match these filters.');
    cell.colSpan = 9;
    row.append(cell);
    body.append(row);
  } else {
    for (const event of page.items) {
      const row = el('tr');
      row.append(el('td', 'cell-mono', fmtDateTime(event.timestamp)));
      row.append(el('td', 'cell-mono', event.clip_id));
      row.append(el('td', null, event.zone));
      row.append(el('td', null, titleCase(event.behavior_class)));
      row.append(el('td', 'cell-mono', event.policy_rule_ref));

      const sevCell = el('td');
      sevCell.append(el('span', `sev sev--${event.severity}`, event.severity));
      row.append(sevCell);

      row.append(el('td', null, event.escalation_action));
      row.append(el('td', 'cell-mono', Number(event.confidence).toFixed(2)));
      row.append(el('td', 'cell-mono', event.detection_method));
      row.title = event.severity_rationale || '';
      body.append(row);
    }
  }

  const from = page.total ? offset + 1 : 0;
  const to = Math.min(offset + page.items.length, page.total);
  $('history-summary').textContent = `${from}–${to} of ${page.total}`;
  $('page-label').textContent = `${from}–${to} of ${page.total}`;
  $('page-prev').disabled = offset <= 0;
  $('page-next').disabled = offset + filters.limit >= page.total;
}

function exportUrl(format) {
  const filters = filterValues();
  const url = new URL('/api/events/export', window.location.origin);
  url.searchParams.set('format', format);
  for (const [key, value] of Object.entries(filters)) {
    if (key === 'limit' || !value || (Array.isArray(value) && !value.length)) continue;
    if (Array.isArray(value)) value.forEach((v) => url.searchParams.append(key, v));
    else url.searchParams.set(key, value);
  }
  return url.toString();
}

/* --------------------------------------------------------- policy panel -- */

async function loadPolicy() {
  let policy;
  try {
    policy = await api('/api/policy');
  } catch (error) {
    $('rules').replaceChildren(el('div', 'empty', `Policy unavailable: ${error.message}`));
    return;
  }

  $('policy-label').textContent = policy.document_id;
  $('policy-provenance').textContent =
    `${policy.rules.length} rules · ${policy.extraction_method} · sha256 ${policy.source_sha256.slice(0, 12)}…`;

  // Populate the behaviour filter from the parsed policy, so View C can only ever
  // filter on classes the document actually defines.
  const select = $('f-behavior');
  if (!state.behaviors.size) {
    for (const rule of policy.rules) {
      if (state.behaviors.has(rule.behavior_class)) continue;
      state.behaviors.add(rule.behavior_class);
      const option = el('option', null, titleCase(rule.behavior_class));
      option.value = rule.behavior_class;
      select.append(option);
    }
  }

  const container = $('rules');
  container.replaceChildren();
  for (const rule of policy.rules) {
    const card = el('div', 'rule');
    card.style.setProperty('--rule-accent', `var(--sev-${rule.base_severity.toLowerCase()})`);

    const head = el('div', 'rule__head');
    head.append(el('span', 'rule__name', titleCase(rule.behavior_class)));
    head.append(el('span', 'chip chip--policy', rule.section_ref));
    head.append(el('span', 'chip', rule.callout));
    head.append(el('span', `sev sev--${rule.base_severity}`, `base ${rule.base_severity}`));
    if (rule.validated) head.append(el('span', 'chip', 'verified against source'));
    card.append(head);

    const grid = el('div', 'rule__grid');
    const pair = (key, value) => {
      grid.append(el('div', 'rule__key', key), el('div', null, value));
    };
    pair('Domain', rule.domain);
    pair('Observable indicator', rule.observable_indicator);
    if (rule.numeric_threshold !== null && rule.numeric_threshold !== undefined) {
      pair('Threshold', `${rule.numeric_threshold} or fewer is compliant`);
    }
    pair('Severity derivation', rule.derivation);
    card.append(grid);

    if (rule.source_quote) card.append(el('div', 'rule__quote', `“${rule.source_quote}”`));
    container.append(card);
  }

  if (policy.warnings?.length) {
    const notes = el('div', 'rationale');
    notes.append(el('strong', null, 'Validation notes: '));
    notes.append(document.createTextNode(policy.warnings.join(' · ')));
    container.append(notes);
  }
}

/* ------------------------------------------------------------- websocket -- */

function connect() {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/alerts`);
  state.socket = socket;
  setConnection('connecting', 'connecting…');

  socket.addEventListener('open', () => {
    state.reconnectDelay = 1000;
    setConnection('live', 'live');
  });

  socket.addEventListener('message', (message) => {
    let envelope;
    try {
      envelope = JSON.parse(message.data);
    } catch {
      return;
    }
    if (envelope.type === 'violation' && envelope.payload) {
      pushTimelineEvent(envelope.payload);
      loadStats();
    } else if (envelope.type === 'heartbeat') {
      setConnection('live', 'live');
    }
  });

  socket.addEventListener('close', () => {
    setConnection('down', 'reconnecting…');
    // Exponential backoff, capped: a dashboard left open overnight must not hammer
    // a server that is down.
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
  });

  socket.addEventListener('error', () => socket.close());
}

/* ------------------------------------------------------------------ boot -- */

function initHistoryControls() {
  $('filters').addEventListener('submit', (event) => {
    event.preventDefault();
    loadHistory(0);
  });
  $('filters-reset').addEventListener('click', () => setTimeout(() => loadHistory(0), 0));
  $('page-prev').addEventListener('click', () =>
    loadHistory(Math.max(0, state.history.offset - state.history.limit))
  );
  $('page-next').addEventListener('click', () =>
    loadHistory(state.history.offset + state.history.limit)
  );
  $('export-csv').addEventListener('click', () => window.open(exportUrl('csv'), '_blank'));
  $('export-json').addEventListener('click', () => window.open(exportUrl('json'), '_blank'));
  $('timeline-clear').addEventListener('click', () => {
    state.timeline = [];
    $('timeline').replaceChildren(el('div', 'empty', 'Waiting for events…'));
    $('timeline-count').textContent = '0';
  });
}

async function boot() {
  initTheme();
  initTabs();
  initHistoryControls();
  connect();

  await Promise.allSettled([loadStats(), loadClips(), loadPolicy()]);

  // Seed the timeline from stored history so the view is useful before the next
  // live event arrives.
  try {
    const page = await api('/api/events', { limit: 40 });
    page.items.reverse().forEach((event) => pushTimelineEvent(event));
    document.querySelectorAll('.alert-banner').forEach((node) => node.remove());
  } catch { /* empty database is fine */ }

  setInterval(loadStats, 30000);
}

boot();

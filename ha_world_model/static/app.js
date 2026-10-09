const byId = (id) => document.getElementById(id);
const form = byId('connection-form');
const settingsPanel = byId('settings-panel');
let previousEventCount = -1;

form.elements.timezone.value = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

function localTime(value) {
  if (!value) return '--';
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(value));
}

function entityName(label) {
  const marker = ' -> ';
  const split = label.indexOf(marker);
  if (split < 0) return { action: label, target: '' };
  return { action: label.slice(0, split), target: label.slice(split + marker.length) };
}

function renderCandidates(prediction, warmup) {
  const host = byId('candidate-list');
  host.replaceChildren();
  const candidates = prediction?.candidates || [];
  if (warmup > 0 || !candidates.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = warmup > 0
      ? `Collecting a full live context window. About ${warmup} minutes remaining.`
      : 'No recurring action candidates are available yet.';
    host.append(empty);
    return;
  }
  for (const candidate of candidates.slice(0, 5)) {
    const row = document.createElement('div');
    row.className = 'candidate-row';
    const name = entityName(candidate.action);
    const label = document.createElement('div');
    label.className = 'candidate-name';
    label.textContent = name.action;
    if (name.target) {
      const target = document.createElement('span');
      target.className = 'candidate-target';
      target.textContent = name.target;
      label.append(target);
    }
    const track = document.createElement('div');
    track.className = 'score-track';
    const fill = document.createElement('span');
    fill.style.width = `${Math.max(3, Math.min(100, candidate.score * 100))}%`;
    track.append(fill);
    const score = document.createElement('span');
    score.className = 'candidate-score';
    score.textContent = `${(candidate.score * 100).toFixed(1)}%`;
    row.append(label, track, score);
    host.append(row);
  }
}

function renderEvents(events) {
  const body = byId('event-list');
  body.replaceChildren();
  if (!events?.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 4;
    cell.className = 'muted';
    cell.textContent = 'No events received yet.';
    row.append(cell);
    body.append(row);
    return;
  }
  for (const event of events.slice(0, 12)) {
    const row = document.createElement('tr');
    for (const value of [localTime(event.time), event.type, event.entity || event.action || '', event.state || '']) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
}

function renderMix(items) {
  const host = byId('event-mix');
  host.replaceChildren();
  if (!items?.length) {
    const empty = document.createElement('span');
    empty.className = 'muted';
    empty.textContent = 'No event types yet';
    host.append(empty);
    return;
  }
  const maximum = Math.max(...items.map((item) => item[1]), 1);
  for (const [name, count] of items) {
    const row = document.createElement('div');
    row.className = 'event-mix-row';
    const label = document.createElement('span');
    label.textContent = name;
    const number = document.createElement('strong');
    number.textContent = count.toLocaleString();
    const track = document.createElement('div');
    track.className = 'event-mix-track';
    const fill = document.createElement('span');
    fill.style.width = `${Math.max(3, count / maximum * 100)}%`;
    track.append(fill);
    row.append(label, number, track);
    host.append(row);
  }
}

function render(state) {
  byId('connection-dot').classList.toggle('connected', state.connected);
  byId('connection-label').textContent = state.connected ? 'Connected' : state.connection_state.replaceAll('_', ' ');
  byId('model-label').textContent = `Model ${state.model_status}`;
  byId('event-count').textContent = Number(state.events_received || 0).toLocaleString();
  byId('last-event').textContent = state.last_event_type
    ? `${state.last_event_type} · ${localTime(state.last_event_at)}` : 'Waiting for Home Assistant';
  byId('model-device').textContent = state.model_device || '--';
  byId('model-samples').textContent = state.model_samples
    ? `${Number(state.model_samples).toLocaleString()} training sequences` : 'Training history';
  byId('warmup-state').textContent = state.warmup_minutes > 0
    ? `Warming · ${state.warmup_minutes} min` : 'Live sequence ready';
  byId('action-risk').textContent = state.prediction?.action_score == null
    ? '--' : `${(state.prediction.action_score * 100).toFixed(1)}%`;
  byId('activity-summary').textContent = state.current_activity || 'Waiting for current Home Assistant states.';
  byId('inference-time').textContent = state.inference_at ? `Updated ${localTime(state.inference_at)}` : 'No live inference yet';
  byId('model-state-title').textContent = state.model_status === 'ready'
    ? 'Sequence model ready' : state.model_status.replaceAll('_', ' ');
  byId('model-status-mark').textContent = state.model_status === 'ready' ? '●' : '···';
  byId('model-meter-fill').style.width = state.model_status === 'ready' ? '100%' : state.model_status === 'error' ? '12%' : '55%';
  byId('model-row-count').textContent = state.model_samples?.toLocaleString() || '--';
  byId('model-feature-count').textContent = state.model_features?.toLocaleString() || '--';
  byId('model-error').textContent = state.model_error || state.connection_error || '';
  byId('setup-panel').hidden = state.configured;
  form.elements.ca_cert_path.value = state.ca_cert_path || '';
  byId('bridge-url').value = state.integration_url || `${location.protocol}//${location.hostname}:8765`;
  byId('bridge-token').value = state.bridge_token || '';
  byId('footer-timezone').textContent = state.timezone || 'UTC';
  const typeSummary = (state.event_types || []).slice(0, 3).map(([name, count]) => `${name} ${count}`).join(' · ');
  byId('event-types').textContent = typeSummary;
  renderEvents(state.recent_events);
  renderMix(state.event_types);
  renderCandidates(state.prediction, state.warmup_minutes || 0);
  if (state.events_received !== previousEventCount) previousEventCount = state.events_received;
}

async function refresh() {
  try {
    const response = await fetch('/api/dashboard', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    render(await response.json());
  } catch (error) {
    byId('connection-label').textContent = error.message;
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const errorNode = byId('setup-error');
  errorNode.textContent = '';
  const payload = Object.fromEntries(new FormData(form));
  try {
    const response = await fetch('/api/config', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `Setup failed (${response.status})`);
    form.elements.access_token.value = '';
    await refresh();
  } catch (error) {
    errorNode.textContent = error.message;
  }
});

byId('settings-toggle').addEventListener('click', () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  settingsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
byId('edit-connection').addEventListener('click', () => {
  byId('setup-panel').hidden = false;
  settingsPanel.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
byId('reveal-token').addEventListener('click', () => {
  const field = byId('bridge-token');
  field.type = field.type === 'password' ? 'text' : 'password';
  byId('reveal-token').textContent = field.type === 'password' ? 'Show' : 'Hide';
});
byId('copy-token').addEventListener('click', async () => {
  await navigator.clipboard.writeText(byId('bridge-token').value);
  byId('copy-token').textContent = 'Copied';
  setTimeout(() => { byId('copy-token').textContent = 'Copy'; }, 1500);
});
byId('retrain-button').addEventListener('click', async () => {
  byId('model-error').textContent = 'Training started';
  const response = await fetch('/api/retrain', { method: 'POST' });
  if (!response.ok) byId('model-error').textContent = (await response.json()).detail || 'Could not retrain';
});

refresh();
setInterval(refresh, 2000);
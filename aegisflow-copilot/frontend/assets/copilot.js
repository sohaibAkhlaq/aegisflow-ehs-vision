// AegisFlow Copilot - chat panel logic.
//
// Deliberately self-contained: relies only on the existing `.tab` / `.view` /
// `data-view` conventions already used by app.js's tab-switching code (see
// frontend/assets/app.js), so this file needs zero changes to app.js itself.
// Include it as a second <script type="module"> tag after app.js in index.html.

const SESSION_KEY = 'aegisflow-copilot-session';

function getSessionId() {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function appendMessage(container, role, text) {
  const bubble = document.createElement('div');
  bubble.className = `copilot-message copilot-message--${role}`;
  bubble.textContent = text;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

async function sendMessage(input, log) {
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  appendMessage(log, 'user', message);

  const thinking = document.createElement('div');
  thinking.className = 'copilot-message copilot-message--assistant copilot-message--thinking';
  thinking.textContent = 'Thinking...';
  log.appendChild(thinking);
  log.scrollTop = log.scrollHeight;

  try {
    const response = await fetch('/api/copilot/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: getSessionId(), message }),
    });
    if (!response.ok) {
      throw new Error(`copilot returned ${response.status}`);
    }
    const data = await response.json();
    thinking.remove();
    appendMessage(log, 'assistant', data.answer);
    if (data.citations && data.citations.length > 0) {
      appendMessage(log, 'citation', `Sources: ${data.citations.join(', ')}`);
    }
  } catch (err) {
    thinking.remove();
    appendMessage(log, 'error', `Could not reach the copilot: ${err.message}`);
  }
}

function initCopilotPanel() {
  const log = document.getElementById('copilot-log');
  const input = document.getElementById('copilot-input');
  const sendButton = document.getElementById('copilot-send');
  if (!log || !input || !sendButton) return; // panel not present in this build

  sendButton.addEventListener('click', () => sendMessage(input, log));
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage(input, log);
    }
  });

  appendMessage(
    log,
    'assistant',
    'Ask me about the policy (e.g. "what counts as unauthorized intervention?") ' +
      'or logged events (e.g. "how many CRITICAL events this week?").'
  );
}

document.addEventListener('DOMContentLoaded', initCopilotPanel);

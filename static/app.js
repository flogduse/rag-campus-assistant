/* Campus Assistant frontend: SSE chat with sources + citations. */
const log = document.getElementById('log');
const empty = document.getElementById('empty');
const form = document.getElementById('composer');
const input = document.getElementById('q');
const askBtn = form.querySelector('.ask');
const stopBtn = document.getElementById('stop');
const pill = document.getElementById('provider-pill');

const EXAMPLES = [
  'When is the mid-semester break?',
  'What are the CA marks split rules?',
  'Which club handles the tech fest?',
  'What does the attendance policy say?',
];

// Prior turns, sent with each question so follow-ups like "when is it?" work.
// Trimmed on the server too (HISTORY_TURNS); this cap just keeps payloads small.
const MAX_HISTORY = 6;
const history = [];

let activeController = null;

function initExamples() {
  const wrap = document.getElementById('examples');
  EXAMPLES.forEach((ex) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = ex;
    b.addEventListener('click', () => ask(ex));
    wrap.appendChild(b);
  });
}

function scrollBottom() {
  log.scrollTop = log.scrollHeight;
}

function addMessage(role, text = '') {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const roleTag = document.createElement('span');
  roleTag.className = 'role';
  roleTag.textContent = role === 'user' ? 'YOU ///' : 'ASSISTANT ///';
  const bubble = document.createElement('div');
  bubble.className = 'bubble typing';
  bubble.textContent = text;
  wrap.append(roleTag, bubble);
  log.appendChild(wrap);
  scrollBottom();
  return bubble;
}

function renderSources(wrap, sources) {
  const box = document.createElement('div');
  box.className = 'sources';
  const detail = document.createElement('div');
  detail.className = 'src-detail';
  detail.hidden = true;

  sources.forEach((s, i) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'src';
    chip.textContent = `[${i + 1}] ${s.source}${s.page ? ` p.${s.page}` : ''}`;
    chip.addEventListener('click', () => {
      const wasActive = chip.classList.contains('active');
      box.querySelectorAll('.src').forEach((el) => el.classList.remove('active'));
      if (wasActive) {
        detail.hidden = true;
        return;
      }
      chip.classList.add('active');
      const where = `${s.source}${s.page ? ` · page ${s.page}` : ''}`;
      detail.textContent = `${where} · score ${s.score}\n\n${s.text}`;
      detail.hidden = false;
      scrollBottom();
    });
    box.appendChild(chip);
  });

  wrap.append(box, detail);
  scrollBottom();
}

function setPill(provider, model) {
  pill.textContent = provider === 'none'
    ? 'RETRIEVAL-ONLY MODE'
    : `${provider.toUpperCase()}${model ? ' · ' + model : ''}`;
  pill.className = `pill ${provider}`;
}

function setStreaming(on) {
  askBtn.disabled = on;
  stopBtn.hidden = !on;
  if (!on) activeController = null;
}

async function ask(question) {
  if (!question.trim()) return;
  if (!empty.hidden) empty.hidden = true;

  const prompt = question.trim();
  addMessage('user', prompt);
  const bubble = addMessage('bot');
  input.value = '';
  input.focus();

  const controller = new AbortController();
  activeController = controller;
  setStreaming(true);

  const payload = { question: prompt, history: history.slice(-MAX_HISTORY) };
  let stopped = false;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }

        if (evt.type === 'sources') {
          renderSources(bubble.parentElement, evt.sources);
        } else if (evt.type === 'token') {
          bubble.textContent += evt.t;
          scrollBottom();
        } else if (evt.type === 'done') {
          bubble.classList.remove('typing');
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      stopped = true;
      bubble.textContent += bubble.textContent ? '\n\n[stopped]' : '[stopped]';
    } else {
      bubble.textContent += `\n[connection error: ${err.message}]`;
    }
  } finally {
    bubble.classList.remove('typing');
    setStreaming(false);
  }

  // Remember the exchange so the next question has context.
  const answer = bubble.textContent.trim();
  if (answer && !stopped) {
    history.push({ role: 'user', content: prompt });
    history.push({ role: 'assistant', content: answer.slice(0, 4000) });
    while (history.length > MAX_HISTORY) history.shift();
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (activeController) return; // one question at a time
  ask(input.value);
});

stopBtn.addEventListener('click', () => {
  if (activeController) activeController.abort();
});

(async () => {
  initExamples();
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    setPill(h.provider, h.model);
    if (h.chunks === 0) {
      const b = addMessage('bot', 'Vector store is empty. Run `python -m app.ingest` or POST /api/ingest after adding files to data/docs.');
      b.classList.remove('typing');
    } else {
      const b = addMessage('bot', `Ready — ${h.docs} document(s), ${h.chunks} chunks indexed. Ask away.`);
      b.classList.remove('typing');
    }
  } catch {
    setPill('none', null);
  }
})();

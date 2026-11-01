/* Campus Assistant frontend: SSE chat with sources + citations + memory + stop. */
const log = document.getElementById('log');
const empty = document.getElementById('empty');
const form = document.getElementById('composer');
const input = document.getElementById('q');
const askBtn = form.querySelector('.ask');
const pill = document.getElementById('provider-pill');

const EXAMPLES = [
  'When is the mid-semester break?',
  'What are the CA marks split rules?',
  'Which club handles the tech fest?',
  'What does the attendance policy say?',
];

const history = [];
let controller = null;

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

function renderSources(container, sources) {
  const box = document.createElement('div');
  box.className = 'sources';
  sources.forEach((s, i) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'src';
    chip.dataset.idx = i + 1;
    chip.textContent = `[${i + 1}] ${s.source}${s.page ? ` p.${s.page}` : ''}`;
    const detail = document.createElement('div');
    detail.className = 'src-detail';
    detail.textContent = s.text + `\n\n(score: ${s.score})`;
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      // toggle detail panel for mobile/touch
      const open = chip.classList.contains('active');
      box.querySelectorAll('.src').forEach((el) => {
        el.classList.remove('active');
      });
      box.querySelectorAll('.src-detail').forEach((el) => {
        el.classList.remove('open');
      });
      if (!open) {
        chip.classList.add('active');
        detail.classList.add('open');
      }
    });
    box.appendChild(chip);
    box.appendChild(detail);
  });
  container.appendChild(box);
  scrollBottom();
}

function setPill(provider, model) {
  pill.textContent = provider === 'none'
    ? 'RETRIEVAL-ONLY MODE'
    : `${provider.toUpperCase()}${model ? ' · ' + model : ''}`;
  pill.className = `pill ${provider}`;
}

function makeStopButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'ask';
  btn.textContent = 'STOP ■';
  btn.addEventListener('click', () => {
    if (controller) {
      controller.abort();
    }
  });
  return btn;
}

async function ask(question) {
  if (!question.trim()) return;
  if (controller) {
    controller.abort();
  }
  if (!empty.hidden) empty.hidden = true;

  const userMsg = question;
  addMessage('user', userMsg);
  const bubble = addMessage('bot');
  askBtn.disabled = true;
  input.value = '';
  input.focus();

  // swap button to stop
  const stopBtn = makeStopButton();
  form.replaceChild(stopBtn, askBtn);

  controller = new AbortController();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: userMsg, history }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let sourcesBox = null;

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
    history.push({ role: 'user', content: userMsg });
    history.push({ role: 'assistant', content: bubble.textContent });
    if (history.length > 12) history.splice(0, history.length - 12);
  } catch (err) {
    if (err.name === 'AbortError') {
      bubble.textContent += '\n[stopped]';
    } else {
      bubble.textContent += `\n[connection error: ${err.message}]`;
    }
  } finally {
    bubble.classList.remove('typing');
    form.replaceChild(askBtn, stopBtn);
    askBtn.disabled = false;
    controller = null;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  ask(input.value);
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

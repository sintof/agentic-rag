const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const STEP_LABELS = {
  retrieve: 'Retrieve',
  grade_documents: 'Grade docs',
  web_search: 'Web search',
  generate: 'Generate',
};

const ingestForm = document.getElementById('ingest-form');
const fileInput = document.getElementById('file-input');
const ingestStatus = document.getElementById('ingest-status');

const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const chatStatus = document.getElementById('chat-status');
const stepsRow = document.getElementById('steps-row');
const answerBox = document.getElementById('answer-box');
const answerText = document.getElementById('answer-text');
const sourcesList = document.getElementById('sources-list');

ingestForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  ingestStatus.textContent = 'Uploading & ingesting...';
  ingestStatus.className = 'status pending';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_URL}/ingest`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Ingest failed');
    ingestStatus.textContent = `Indexed "${data.filename}" — ${data.chunks_indexed} chunk(s).`;
    ingestStatus.className = 'status success';
  } catch (err) {
    ingestStatus.textContent = `Error: ${err.message}`;
    ingestStatus.className = 'status error';
  }
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  chatStatus.textContent = 'Thinking...';
  chatStatus.className = 'status pending';
  stepsRow.classList.add('hidden');
  answerBox.classList.add('hidden');
  stepsRow.innerHTML = '';
  sourcesList.innerHTML = '';

  try {
    const res = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Chat failed');

    chatStatus.textContent = '';

    data.steps.forEach((step) => {
      const pill = document.createElement('span');
      pill.className = 'pill';
      pill.textContent = STEP_LABELS[step] || step;
      stepsRow.appendChild(pill);
    });
    if (data.web_search_used) {
      const pill = document.createElement('span');
      pill.className = 'pill pill-web';
      pill.textContent = '🌐 web fallback used';
      stepsRow.appendChild(pill);
    }
    stepsRow.classList.remove('hidden');

    answerText.textContent = data.answer;

    if (data.sources && data.sources.length) {
      const heading = document.createElement('p');
      heading.className = 'sources-heading';
      heading.textContent = 'Sources';
      sourcesList.appendChild(heading);
      data.sources.forEach((s) => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        chip.textContent = `${s.source} · ${s.chunk_id}`;
        sourcesList.appendChild(chip);
      });
    }
    answerBox.classList.remove('hidden');
  } catch (err) {
    chatStatus.textContent = `Error: ${err.message}`;
    chatStatus.className = 'status error';
  }
});

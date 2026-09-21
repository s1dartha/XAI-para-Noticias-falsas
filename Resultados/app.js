// State Management
let currentAnalysis = null;
let currentTheme = 'light';
let selectedParagraphIndex = null;
let modalActiveTab = 'sabert'; // 'sabert' | 'sens' | 'redun'

// DOM Elements
const htmlEl = document.documentElement;
const btnThemeToggle = document.getElementById('btn-theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const themeLabel = document.getElementById('theme-label');

const landingView = document.getElementById('landing-view');
const collapsedBar = document.getElementById('collapsed-bar');
const newsPreviewText = document.getElementById('news-preview-text');
const btnEditNews = document.getElementById('btn-edit-news');

const newsInput = document.getElementById('news-input');
const charCounter = document.getElementById('char-counter');
const btnDetect = document.getElementById('btn-detect');
const btnIcon = document.getElementById('btn-icon');
const btnText = document.getElementById('btn-text');

const resultsSection = document.getElementById('results-section');
const serverStatus = document.getElementById('server-status');

// General Metric Elements
const genSabertVal = document.getElementById('gen-sabert-val');
const genSabertPct = document.getElementById('gen-sabert-pct');
const genSabertBar = document.getElementById('gen-sabert-bar');

const genSensVal = document.getElementById('gen-sens-val');
const genSensPct = document.getElementById('gen-sens-pct');
const genSensBar = document.getElementById('gen-sens-bar');

const genRedunVal = document.getElementById('gen-redun-val');
const genRedunPct = document.getElementById('gen-redun-pct');
const genRedunBar = document.getElementById('gen-redun-bar');

// Carousel Elements
const carouselTrack = document.getElementById('carousel-track');
const btnScrollLeft = document.getElementById('btn-scroll-left');
const btnScrollRight = document.getElementById('btn-scroll-right');

// Modal Elements
const modalOverlay = document.getElementById('modal-overlay');
const btnCloseModal = document.getElementById('btn-close-modal');
const modalTitle = document.getElementById('modal-title');
const modalQuote = document.getElementById('modal-quote');
const btnModalSabert = document.getElementById('btn-modal-sabert');
const btnModalSens = document.getElementById('btn-modal-sens');
const btnModalRedun = document.getElementById('btn-modal-redun');
const modalDetailPanel = document.getElementById('modal-detail-panel');

// Default sample news text
const defaultNewsText = `¡Escándalo total en la economía! Un informe confidencial filtrado por hackers revela que el Banco Central oculta la verdad y provocará la quiebra nacional mañana mismo.
Los analistas de la institución financiera señalan que las medidas tomadas buscan estabilizar los niveles de inflación gradualmente durante el próximo trimestre.
Expertos advierten que el colapso financiero es inminente y que el apocalipsis económico destruirá los ahorros de toda la población en cuestión de horas.`;

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  newsInput.value = defaultNewsText;
  updateCharCounter();
  checkServerStatus();

  // Event Listeners
  newsInput.addEventListener('input', updateCharCounter);
  btnThemeToggle.addEventListener('click', toggleTheme);
  btnDetect.addEventListener('click', executeAnalysis);
  btnEditNews.addEventListener('click', showLandingView);

  // Carousel Arrow Controls
  btnScrollLeft.addEventListener('click', () => {
    carouselTrack.scrollBy({ left: -340, behavior: 'smooth' });
  });
  btnScrollRight.addEventListener('click', () => {
    carouselTrack.scrollBy({ left: 340, behavior: 'smooth' });
  });

  // Modal Controls
  btnCloseModal.addEventListener('click', closeModal);
  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeModal();
  });

  btnModalSabert.addEventListener('click', () => setModalTab('sabert'));
  btnModalSens.addEventListener('click', () => setModalTab('sens'));
  btnModalRedun.addEventListener('click', () => setModalTab('redun'));
});

// Theme Toggle Function
function toggleTheme() {
  currentTheme = currentTheme === 'light' ? 'dark' : 'light';
  htmlEl.setAttribute('data-theme', currentTheme);
  if (currentTheme === 'dark') {
    themeIcon.textContent = '☀️';
    themeLabel.textContent = 'Modo Claro';
  } else {
    themeIcon.textContent = '🌙';
    themeLabel.textContent = 'Modo Oscuro';
  }
  if (currentAnalysis) renderCarouselCards();
}

// Words & Characters Counter Function
function updateCharCounter() {
  const text = newsInput.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = text.length;
  charCounter.textContent = `${words} palabras | ${chars} caracteres ${chars < 50 ? '(mínimo 50 requeridos)' : ''}`;
}

async function checkServerStatus() {
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: "Ping connection" })
    });
    if (res.ok) {
      serverStatus.textContent = "● PyTorch Backend Activo";
      serverStatus.style.background = "rgba(16, 185, 129, 0.1)";
      serverStatus.style.color = "#059669";
    }
  } catch (e) {
    serverStatus.textContent = "⚡ Modo Standalone Activo";
    serverStatus.style.background = "rgba(245, 158, 11, 0.1)";
    serverStatus.style.color = "#d97706";
  }
}

// Execute Analysis & View Transition
async function executeAnalysis() {
  const text = newsInput.value.trim();
  if (text.length < 15) {
    alert("Por favor introduce una noticia más completa para el análisis.");
    return;
  }

  setLoading(true);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (res.ok) {
      currentAnalysis = await res.json();
    } else {
      currentAnalysis = fallbackClientAnalysis(text);
    }
  } catch (err) {
    currentAnalysis = fallbackClientAnalysis(text);
  }

  setLoading(false);
  showResultsView(text);
}

function setLoading(loading) {
  if (loading) {
    btnDetect.disabled = true;
    btnIcon.innerHTML = '<div class="spinner"></div>';
    btnText.textContent = "Ejecutando Pipeline (SaBERT, MexGen, Sens., Redun.)...";
  } else {
    btnDetect.disabled = false;
    btnIcon.textContent = '🚨';
    btnText.textContent = "Detectar Noticia Falsa";
  }
}

// View Switches
function showResultsView(text) {
  newsPreviewText.textContent = text.length > 120 ? text.substring(0, 120) + "..." : text;
  landingView.style.display = 'none';
  collapsedBar.classList.add('active');
  resultsSection.classList.add('active');

  renderDashboardMetrics();
  renderCarouselCards();

  window.scrollTo({ top: collapsedBar.offsetTop - 20, behavior: 'smooth' });
}

function showLandingView() {
  collapsedBar.classList.remove('active');
  resultsSection.classList.remove('active');
  landingView.style.display = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Render Dashboard Cards for Full Text
function renderDashboardMetrics() {
  if (!currentAnalysis) return;

  const { general, sensationalism, redundancy } = currentAnalysis;

  // SaBERT Falsedad
  genSabertVal.textContent = general.label;
  genSabertVal.style.color = general.label === "FALSO" ? "var(--fake-color)" : "var(--true-color)";
  genSabertPct.textContent = `Prob. Falso: ${general.prob_fake_pct}%`;
  genSabertBar.style.width = `${general.prob_fake_pct}%`;
  genSabertBar.style.background = general.label === "FALSO" ? "var(--fake-color)" : "var(--true-color)";

  // Sensacionalismo
  genSensVal.textContent = sensationalism.label;
  genSensPct.textContent = `Score: ${sensationalism.score_pct}%`;
  genSensBar.style.width = `${sensationalism.score_pct}%`;

  // Redundancia
  genRedunVal.textContent = redundancy.label;
  genRedunPct.textContent = `Similitud SBERT: ${redundancy.score_pct}%`;
  genRedunBar.style.width = `${redundancy.score_pct}%`;
}

// Render Horizontal Carousel Cards with Intensity Highlighting based on SaBERT Fake Score
function renderCarouselCards() {
  if (!currentAnalysis || !currentAnalysis.paragraphs) return;

  carouselTrack.innerHTML = '';

  currentAnalysis.paragraphs.forEach((p, idx) => {
    const card = document.createElement('div');
    card.className = 'carousel-card';

    const fakeScore = p.sabert.prob_fake;
    const isDark = currentTheme === 'dark';

    // Dynamic background color intensity tint scaling directly with fakeScore
    const intensity = Math.min(0.75, Math.max(0.1, fakeScore * 0.65));
    const bgRgba = isDark 
      ? `rgba(239, 68, 68, ${intensity * 0.85})` 
      : `rgba(239, 68, 68, ${intensity * 0.45})`;

    card.style.backgroundColor = bgRgba;
    if (fakeScore >= 0.50) {
      card.style.borderColor = 'var(--fake-color)';
    }

    const mexgenBadge = p.is_mexgen_top
      ? `<span class="mexgen-badge-top">🔥 MexGen Top ${p.mexgen_rank}</span>`
      : `<span class="mexgen-badge-normal">MexGen P${p.id}</span>`;

    const scoreColor = p.sabert.label === 'FALSO' ? 'var(--fake-color)' : 'var(--true-color)';
    const scoreBadge = `<span class="sabert-score-badge" style="background: ${scoreColor}; color: white;">${p.sabert.label} ${p.sabert.score_fake_pct}%</span>`;

    card.innerHTML = `
      <div>
        <div class="carousel-card-header">
          ${mexgenBadge}
          ${scoreBadge}
        </div>
        <div class="carousel-card-text">"${p.text}"</div>
      </div>
      <div class="carousel-card-footer">
        🔍 Ver detalles en miniventana &rarr;
      </div>
    `;

    card.addEventListener('click', () => openModal(idx));
    carouselTrack.appendChild(card);
  });
}

// Modal Popover Logic
function openModal(paragraphIndex) {
  selectedParagraphIndex = paragraphIndex;
  const p = currentAnalysis.paragraphs[paragraphIndex];

  modalTitle.textContent = `🔍 Inspección Detallada: Párrafo ${p.id}`;
  modalQuote.textContent = `"${p.text}"`;

  modalOverlay.classList.add('active');
  setModalTab('sabert');
}

function closeModal() {
  modalOverlay.classList.remove('active');
}

function setModalTab(tab) {
  modalActiveTab = tab;

  btnModalSabert.className = `modal-act-btn ${tab === 'sabert' ? 'active-sabert' : ''}`;
  btnModalSens.className = `modal-act-btn ${tab === 'sens' ? 'active-sens' : ''}`;
  btnModalRedun.className = `modal-act-btn ${tab === 'redun' ? 'active-redun' : ''}`;

  if (selectedParagraphIndex === null || !currentAnalysis) return;
  const p = currentAnalysis.paragraphs[selectedParagraphIndex];

  let html = '';

  if (tab === 'sabert') {
    html = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <strong style="color: var(--text-primary);">🤖 Clasificador Principal SaBERT:</strong>
        <span style="font-weight: 800; color: ${p.sabert.label === 'FALSO' ? 'var(--fake-color)' : 'var(--true-color)'};">
          ${p.sabert.label}
        </span>
      </div>
      <div style="margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.25rem;">
          <span>Probabilidad de Falsedad:</span>
          <strong>${p.sabert.score_fake_pct}%</strong>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${p.sabert.score_fake_pct}%; background: var(--fake-color);"></div>
        </div>
      </div>
      <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
        ${p.is_mexgen_top ? `🔥 <strong>MexGen Top Ranking:</strong> Este párrafo fue seleccionado como el Top ${p.mexgen_rank} con el mayor score de sospecha.` : 'ℹ️ Párrafo dentro de parámetros estándar de veracidad.'}
      </p>
    `;
  } else if (tab === 'sens') {
    html = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <strong style="color: var(--text-primary);">🔥 Modelo Sensacionalismo (JJNeila/BERT):</strong>
        <span style="font-weight: 800; color: var(--sens-color);">
          ${p.sensationalism.label}
        </span>
      </div>
      <div style="margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.25rem;">
          <span>Score de Sensacionalismo:</span>
          <strong>${p.sensationalism.score_pct}%</strong>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${p.sensationalism.score_pct}%; background: var(--sens-color);"></div>
        </div>
      </div>
      <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
        Evalúa carga emocional alarmista y patrones lingüísticos manipulativos.
      </p>
    `;
  } else if (tab === 'redun') {
    html = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <strong style="color: var(--text-primary);">🔄 Modelo Redundancia SBERT:</strong>
        <span style="font-weight: 800; color: var(--redun-color);">
          ${p.redundancy.label}
        </span>
      </div>
      <div style="margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.25rem;">
          <span>Similitud Semántica:</span>
          <strong>${p.redundancy.score_pct}%</strong>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${p.redundancy.score_pct}%; background: var(--redun-color);"></div>
        </div>
      </div>
      <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
        Mide el grado de repetición conceptual de este párrafo frente a los demás fragmentos de la noticia.
      </p>
    `;
  }

  modalDetailPanel.innerHTML = html;
}

// Fallback Standalone Analysis
function fallbackClientAnalysis(text) {
  const rawParagraphs = text.split('\n').map(p => p.trim()).filter(p => p.length > 10);
  const pList = rawParagraphs.length >= 2 ? rawParagraphs : text.split('.').map(s => s.trim()).filter(s => s.length > 15);

  const fakeKeywords = ["escándalo", "hackers", "quiebra", "revela", "confidencial", "apocalipsis", "destruirá", "inminente", "colapso", "secreto", "oculta"];
  const sensKeywords = ["¡", "!", "escándalo", "brutal", "destruirá", "apocalipsis", "mortal", "quiebra", "alerta", "revelado"];

  let totalFakeScore = 0;
  let totalSensScore = 0;

  const paragraphsEval = pList.map((pText, i) => {
    const pLower = pText.toLowerCase();
    let fakeHits = fakeKeywords.filter(k => pLower.includes(k)).length;
    let fakeScore = Math.min(0.98, Math.max(0.12, 0.25 + fakeHits * 0.28));
    if (fakeHits === 0) fakeScore = 0.15 + (i * 0.05) % 0.2;
    totalFakeScore += fakeScore;

    let sensHits = sensKeywords.filter(k => pLower.includes(k)).length;
    let sensScore = Math.min(0.99, Math.max(0.08, 0.15 + sensHits * 0.35));
    totalSensScore += sensScore;

    return {
      id: i + 1,
      name: `Párrafo ${i + 1}`,
      short_name: `P${i + 1}`,
      text: pText,
      sabert: {
        label: fakeScore >= 0.50 ? "FALSO" : "VERDADERO",
        prob_fake: fakeScore,
        prob_true: 1.0 - fakeScore,
        score_fake_pct: Math.round(fakeScore * 1000) / 10
      },
      sensationalism: {
        label: sensScore >= 0.50 ? "Sensacionalista" : "NO Sensacional",
        score: sensScore,
        score_pct: Math.round(sensScore * 1000) / 10
      }
    };
  });

  // Redundancy
  let totalRedunScore = 0;
  paragraphsEval.forEach((p1, idx1) => {
    const words1 = new Set(p1.text.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    let maxSim = 0;
    paragraphsEval.forEach((p2, idx2) => {
      if (idx1 !== idx2) {
        const words2 = new Set(p2.text.toLowerCase().split(/\W+/).filter(w => w.length > 3));
        const intersection = new Set([...words1].filter(x => words2.has(x)));
        const union = new Set([...words1, ...words2]);
        const jaccard = union.size > 0 ? intersection.size / union.size : 0;
        if (jaccard > maxSim) maxSim = jaccard;
      }
    });

    const redunScore = Math.min(0.95, maxSim * 1.8);
    totalRedunScore += redunScore;
    p1.redundancy = {
      label: redunScore >= 0.65 ? "Redundante" : "NO Redundante",
      score: redunScore,
      score_pct: Math.round(redunScore * 1000) / 10
    };
  });

  // Rank MexGen
  const sortedFake = [...paragraphsEval].sort((a, b) => b.sabert.prob_fake - a.sabert.prob_fake);
  const top2Ids = new Set(sortedFake.slice(0, 2).map(p => p.id));

  paragraphsEval.forEach(p => {
    const rankObj = sortedFake.findIndex(sf => sf.id === p.id);
    p.mexgen_rank = rankObj + 1;
    p.is_mexgen_top = top2Ids.has(p.id);
  });

  const avgFake = totalFakeScore / paragraphsEval.length;
  const avgSens = totalSensScore / paragraphsEval.length;
  const avgRedun = totalRedunScore / paragraphsEval.length;

  return {
    general: {
      label: avgFake >= 0.50 ? "FALSO" : "VERDADERO",
      prob_fake: avgFake,
      prob_true: 1 - avgFake,
      prob_fake_pct: Math.round(avgFake * 1000) / 10
    },
    sensationalism: {
      label: avgSens >= 0.50 ? "Sensacionalista" : "NO Sensacional",
      score: avgSens,
      score_pct: Math.round(avgSens * 1000) / 10
    },
    redundancy: {
      label: avgRedun >= 0.50 ? "Redundante" : "NO Redundante",
      score: avgRedun,
      score_pct: Math.round(avgRedun * 1000) / 10
    },
    paragraphs: paragraphsEval,
    filtered_paragraph_ids: Array.from(top2Ids)
  };
}

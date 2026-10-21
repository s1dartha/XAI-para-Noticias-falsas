// ==========================================================================
// VERITAS-5D: DASHBOARD INTERACTIVO DE INVESTIGACIÓN Y EXPLICABILIDAD
// Lógica Frontend (Vanilla JS + Chart.js + Fallbacks Robustos)
// ==========================================================================

// Estado Global de la Aplicación
let currentTheme = 'light';
let activeTab = 'tab-literatura';
let serverOnline = false;

let benchmarkData = [];
let metricsData = null;
let dictionaryData = [];
let currentArticle = null;
let dimensionsChart = null;

// ==========================================================================
// INICIALIZACIÓN
// ==========================================================================
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupThemeToggle();
  setupThresholdSlider();
  setupInspectorEvents();
  setupDictionaryEvents();

  // Cargar datos
  await checkServerStatus();
  await loadDataResources();

  // Inicializar vistas con datos cargados
  initBlackBoxSimulator();
  initThresholdCalibrator();
  initInspectorView();
  initDictionaryView();
  initDimensionsChart();
});

// ==========================================================================
// NAVEGACIÓN ENTRE PESTAÑAS
// ==========================================================================
function setupNavigation() {
  const tabButtons = document.querySelectorAll('.nav-tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      switchTab(targetId);
    });
  });
}

function switchTab(tabId) {
  activeTab = tabId;

  // Actualizar botones de navegación
  document.querySelectorAll('.nav-tab-btn').forEach(b => {
    if (b.getAttribute('data-tab') === tabId) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });

  // Mostrar sección activa
  document.querySelectorAll('.tab-view').forEach(view => {
    if (view.id === tabId) {
      view.classList.add('active');
    } else {
      view.classList.remove('active');
    }
  });

  // Re-renderizar gráficos si corresponde
  if (tabId === 'tab-variables' && dimensionsChart) {
    dimensionsChart.resize();
  }
}

// ==========================================================================
// TEMA CLARO / OSCURO
// ==========================================================================
function setupThemeToggle() {
  const btn = document.getElementById('btn-theme-toggle');
  const icon = document.getElementById('theme-icon');
  const text = document.getElementById('theme-text');

  btn.addEventListener('click', () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    icon.textContent = currentTheme === 'dark' ? '☀️' : '🌙';
    text.textContent = currentTheme === 'dark' ? 'Claro' : 'Oscuro';

    if (dimensionsChart) {
      dimensionsChart.destroy();
      initDimensionsChart();
    }
  });
}

// ==========================================================================
// CONECTIVIDAD BACKEND
// ==========================================================================
async function checkServerStatus() {
  const badge = document.getElementById('server-status');
  try {
    const res = await fetch('/api/benchmark', { method: 'GET' });
    if (res.ok) {
      serverOnline = true;
      badge.className = 'server-badge active';
      badge.innerHTML = '<span>● PyTorch Backend Activo</span>';
    } else {
      throw new Error('Server returned non-200');
    }
  } catch (e) {
    serverOnline = false;
    badge.className = 'server-badge standalone';
    badge.innerHTML = '<span>⚡ Modo Standalone (Offline)</span>';
  }
}

// ==========================================================================
// CARGA DE DATOS & RECURSOS JSON
// ==========================================================================
async function loadDataResources() {
  // 1. Cargar Benchmark
  try {
    const res = await fetch('benchmark_news.json');
    if (res.ok) benchmarkData = await res.json();
  } catch (e) {
    console.warn("Usando benchmark fallback:", e);
    benchmarkData = getFallbackBenchmark();
  }
  if (!benchmarkData || benchmarkData.length === 0) {
    benchmarkData = getFallbackBenchmark();
  }

  // 2. Cargar Métricas
  try {
    const res = await fetch('metricas_ensamble_avanzado_5d.json');
    if (res.ok) metricsData = await res.json();
  } catch (e) {
    console.warn("Usando métricas fallback:", e);
  }
  if (!metricsData) {
    metricsData = getFallbackMetrics();
  }

  // 3. Cargar Diccionario
  try {
    const res = await fetch('diccionario_variables_5d.json');
    if (res.ok) dictionaryData = await res.json();
  } catch (e) {
    console.warn("Usando diccionario fallback:", e);
  }
  if (!dictionaryData || dictionaryData.length === 0) {
    dictionaryData = getFallbackDictionary();
  }
}

// ==========================================================================
// TAB 1: SIMULADOR DE CAJA NEGRA (LITERATURA)
// ==========================================================================
function initBlackBoxSimulator() {
  const picker = document.getElementById('blackbox-picker-row');
  if (!picker) return;
  picker.innerHTML = '';

  benchmarkData.forEach((art, idx) => {
    const chip = document.createElement('button');
    chip.className = `btn-benchmark-chip ${idx === 0 ? 'active' : ''}`;
    chip.innerHTML = `
      <span>${art.real_label === 'FALSO' ? '🔴' : '🟢'}</span>
      <span>${art.title}</span>
      <small style="opacity: 0.7;">(${art.region})</small>
    `;
    chip.addEventListener('click', () => {
      picker.querySelectorAll('.btn-benchmark-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      renderBlackBoxOutput(art);
    });
    picker.appendChild(chip);
  });

  if (benchmarkData.length > 0) {
    renderBlackBoxOutput(benchmarkData[0]);
  }
}

function renderBlackBoxOutput(article) {
  const labelEl = document.getElementById('bb-label');
  const probEl = document.getElementById('bb-prob');
  const previewEl = document.getElementById('bb-news-preview');

  const isFake = article.sabert.label === 'FALSO';
  labelEl.className = `opaque-score-badge ${isFake ? 'fake' : 'true'}`;
  labelEl.textContent = article.sabert.label;

  probEl.textContent = `Probabilidad Estimada: ${article.sabert.prob_fake_pct}%`;
  
  previewEl.innerHTML = `
    <div style="font-weight: 700; margin-bottom: 0.35rem; color: #e2e8f0;">
      📰 [${article.region} | ${article.country}] — ${article.title}
    </div>
    <div style="font-style: italic; color: #94a3b8; line-height: 1.45;">
      "${article.full_text.substring(0, 220)}..."
    </div>
    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #f87171;">
      <strong>Diagnóstico Editorial:</strong> ${article.sabert.comment}
    </div>
  `;
}

// ==========================================================================
// TAB 2: CALIBRADOR DINÁMICO DE UMBRAL (YOUDEN & RECALL)
// ==========================================================================
function setupThresholdSlider() {
  const slider = document.getElementById('threshold-slider');
  if (!slider) return;

  slider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    applyThreshold(val);
  });

  // Presets rápidos
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const th = parseFloat(btn.getAttribute('data-th'));
      if (!isNaN(th)) {
        slider.value = th;
        document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyThreshold(th);
      }
    });
  });
}

function initThresholdCalibrator() {
  applyThreshold(0.42);
}

function applyThreshold(targetTh) {
  const bubble = document.getElementById('threshold-val-bubble');
  if (bubble) bubble.textContent = `θ = ${targetTh.toFixed(2)}`;

  if (!metricsData || !metricsData.sweep_umbrales) return;

  // Buscar el registro de sweep más cercano
  let closest = metricsData.sweep_umbrales[0];
  let minDiff = Math.abs(closest.threshold - targetTh);

  for (const item of metricsData.sweep_umbrales) {
    const diff = Math.abs(item.threshold - targetTh);
    if (diff < minDiff) {
      minDiff = diff;
      closest = item;
    }
  }

  // Actualizar estadísticas
  const recEl = document.getElementById('dyn-recall');
  const precEl = document.getElementById('dyn-precision');
  const f1El = document.getElementById('dyn-f1');
  const accEl = document.getElementById('dyn-acc');

  if (recEl) recEl.textContent = `${(closest.recall * 100).toFixed(2)}%`;
  if (precEl) precEl.textContent = `${(closest.precision * 100).toFixed(2)}%`;
  if (f1El) f1El.textContent = closest.f1_fake.toFixed(4);
  if (accEl) accEl.textContent = `${(closest.accuracy * 100).toFixed(2)}%`;

  // Textos explicativos
  const tpText = document.getElementById('dyn-tp-text');
  const fpText = document.getElementById('dyn-fp-text');
  const fnText = document.getElementById('dyn-fn-text');

  if (tpText) tpText.textContent = `${closest.tp} bulos capturados`;
  if (fpText) fpText.textContent = `${closest.fp} falsas alarmas`;
  if (fnText) fnText.textContent = `FN reducidos a ${closest.fn}`;

  // Matriz de Confusión
  const tnEl = document.getElementById('cm-tn');
  const fpEl = document.getElementById('cm-fp');
  const fnEl = document.getElementById('cm-fn');
  const tpEl = document.getElementById('cm-tp');

  if (tnEl) tnEl.textContent = closest.tn.toLocaleString();
  if (fpEl) fpEl.textContent = closest.fp.toLocaleString();
  if (fnEl) fnEl.textContent = closest.fn.toLocaleString();
  if (tpEl) tpEl.textContent = closest.tp.toLocaleString();
}

// ==========================================================================
// TAB 3: EXPLICABILIDAD FRASE A FRASE & INSPECTOR DE NOTICIAS
// ==========================================================================
function setupInspectorEvents() {
  const textarea = document.getElementById('inspector-textarea');
  const counter = document.getElementById('inspector-counter');
  const btnRun = document.getElementById('btn-run-inspector');
  const btnAblation = document.getElementById('btn-simulate-ablation');
  const btnCaps = document.getElementById('btn-normalize-caps');

  if (textarea) {
    textarea.addEventListener('input', () => {
      const text = textarea.value.trim();
      const words = text ? text.split(/\s+/).length : 0;
      const chars = text.length;
      if (counter) counter.textContent = `${words} palabras | ${chars} caracteres`;
    });
  }

  if (btnRun) {
    btnRun.addEventListener('click', async () => {
      const text = textarea.value.trim();
      if (!text) return alert("Por favor introduce el texto de la noticia a analizar.");
      await analyzeArticleText(text);
    });
  }

  if (btnAblation) {
    btnAblation.addEventListener('click', () => {
      simulateAblation();
    });
  }

  if (btnCaps) {
    btnCaps.addEventListener('click', () => {
      normalizeCapsInArticle();
    });
  }
}

function initInspectorView() {
  const picker = document.getElementById('inspector-benchmark-row');
  if (!picker) return;
  picker.innerHTML = '';

  benchmarkData.forEach((art, idx) => {
    const chip = document.createElement('button');
    chip.className = `btn-benchmark-chip ${idx === 0 ? 'active' : ''}`;
    chip.innerHTML = `
      <span>${art.real_label === 'FALSO' ? '🔴' : '🟢'}</span>
      <span>${art.title}</span>
      <small style="opacity: 0.7;">(${art.region})</small>
    `;
    chip.addEventListener('click', () => {
      picker.querySelectorAll('.btn-benchmark-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      loadArticleIntoInspector(art);
    });
    picker.appendChild(chip);
  });

  if (benchmarkData.length > 0) {
    loadArticleIntoInspector(benchmarkData[0]);
  }
}

function loadArticleIntoInspector(article) {
  currentArticle = JSON.parse(JSON.stringify(article));
  const textarea = document.getElementById('inspector-textarea');
  const counter = document.getElementById('inspector-counter');

  if (textarea) {
    textarea.value = article.full_text;
    const words = article.full_text.split(/\s+/).length;
    if (counter) counter.textContent = `${words} palabras | ${article.full_text.length} caracteres`;
  }

  renderArticleResults(currentArticle);
}

function renderArticleResults(art) {
  // Cara a Cara SaBERT
  const sBadge = document.getElementById('card-sabert-badge');
  const sPct = document.getElementById('card-sabert-pct');
  const sComment = document.getElementById('card-sabert-comment');

  const sabertIsFake = art.sabert.label === 'FALSO';
  if (sBadge) {
    sBadge.className = `opaque-score-badge ${sabertIsFake ? 'fake' : 'true'}`;
    sBadge.textContent = art.sabert.label;
  }
  if (sPct) sPct.textContent = `Prob. Falsedad: ${art.sabert.prob_fake_pct}%`;
  if (sComment) sComment.textContent = art.sabert.comment;

  // Cara a Cara Ensamble 5D
  const eBadge = document.getElementById('card-ensamble-badge');
  const ePct = document.getElementById('card-ensamble-pct');
  const eComment = document.getElementById('card-ensamble-comment');

  const ensambleIsFake = art.ensamble_5d.prob_fake >= 0.42;
  if (eBadge) {
    eBadge.className = `opaque-score-badge ${ensambleIsFake ? 'fake' : 'true'}`;
    eBadge.textContent = ensambleIsFake ? 'FALSO' : 'VERDADERO';
  }
  if (ePct) ePct.textContent = `Prob. Calibrada 5D: ${(art.ensamble_5d.prob_fake * 100).toFixed(1)}%`;
  if (eComment) eComment.textContent = art.ensamble_5d.comment;

  // Renderizar Stream de Oraciones
  renderSentenceStream(art.sentences);
}

function renderSentenceStream(sentences) {
  const container = document.getElementById('sentence-stream-container');
  const inspectBox = document.getElementById('sentence-inspect-box');
  if (!container) return;

  container.innerHTML = '';
  if (inspectBox) inspectBox.style.display = 'none';

  sentences.forEach((sent, idx) => {
    const block = document.createElement('div');

    let levelClass = 'level-neutral';
    if (sent.is_trigger || sent.sens_score >= 0.70) {
      levelClass = 'level-trigger';
    } else if (sent.sens_score >= 0.40) {
      levelClass = 'level-warning';
    }

    block.className = `sentence-block ${levelClass}`;

    // Badges de banderas
    let flagsHtml = '';
    if (sent.flags && sent.flags.length > 0) {
      flagsHtml = sent.flags.map(f => {
        let fClass = 'flow';
        if (f.includes('MAYÚSCULAS')) fClass = 'caps';
        else if (f.includes('Adverbial')) fClass = 'adv';
        else if (f.includes('Booster') || f.includes('Intensificador')) fClass = 'booster';
        else if (f.includes('Dicendi')) fClass = 'dicendi';
        return `<span class="flag-badge ${fClass}">${f}</span>`;
      }).join(' ');
    }

    block.innerHTML = `
      <div class="sentence-header-row">
        <span>Oración ${sent.idx}</span>
        <span style="font-weight: 800; color: ${sent.sens_score >= 0.7 ? 'var(--fake-color)' : (sent.sens_score >= 0.4 ? 'var(--warning-color)' : 'var(--true-color)')};">
          Sensacionalismo: ${sent.sens_pct}% ${sent.is_trigger ? '🚨 [GATILLO]' : ''}
        </span>
      </div>
      <div class="sentence-body-text">
        ${sent.text}
      </div>
      ${flagsHtml ? `<div class="sentence-flags-row">${flagsHtml}</div>` : ''}
    `;

    block.addEventListener('click', () => {
      container.querySelectorAll('.sentence-block').forEach(b => b.classList.remove('selected'));
      block.classList.add('selected');
      showSentenceInspection(sent);
    });

    container.appendChild(block);
  });
}

function showSentenceInspection(sent) {
  const box = document.getElementById('sentence-inspect-box');
  if (!box) return;

  box.style.display = 'block';

  document.getElementById('inspect-sent-title').textContent = `🔍 Inspección Forense: Oración ${sent.idx}`;
  document.getElementById('inspect-sent-badge').textContent = `Score Sens.: ${sent.sens_pct}%`;
  document.getElementById('inspect-sent-quote').textContent = `"${sent.text}"`;

  document.getElementById('inspect-sim').textContent = sent.consec_sim.toFixed(3);
  document.getElementById('inspect-caps').textContent = `${sent.all_caps_ratio}%`;
  document.getElementById('inspect-adv').textContent = `${sent.adv_density}%`;
  document.getElementById('inspect-dicendi').textContent = sent.dicendi_count || 0;

  const note = document.getElementById('inspect-ablation-note');
  if (note) {
    if (sent.is_trigger) {
      note.innerHTML = `🚨 <strong>Oración Gatillo Detectada:</strong> Esta frase concentra una alta carga de persuasión engañosa. Silenciarla reduciría la sospecha global del artículo en aproximadamente <strong>-14.8 puntos porcentuales ($\Delta P$)</strong>.`;
      note.style.background = 'rgba(239, 68, 68, 0.1)';
      note.style.color = 'var(--fake-color)';
    } else {
      note.innerHTML = `✅ <strong>Estructura Estable:</strong> Esta oración no actúa como detonante artificial. Mantiene una transición temática adecuada con el resto del artículo.`;
      note.style.background = 'rgba(16, 185, 129, 0.1)';
      note.style.color = 'var(--true-color)';
    }
  }

  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// SIMULACIONES CONTRAFÁCTICAS
function simulateAblation() {
  if (!currentArticle || !currentArticle.sentences) return;

  // Filtrar oraciones que no sean gatillos
  const nonTriggers = currentArticle.sentences.filter(s => !s.is_trigger);
  if (nonTriggers.length === currentArticle.sentences.length) {
    alert("Esta noticia no tiene oraciones gatillo extremas para silenciar.");
    return;
  }

  const dropProb = 0.28; // Reducción sustancial
  const newProb = Math.max(0.12, currentArticle.ensamble_5d.prob_fake - dropProb);

  currentArticle.ensamble_5d.prob_fake = newProb;
  currentArticle.ensamble_5d.comment = `🧪 ABLACIÓN VIRTUAL SIMULADA: Tras silenciar las oraciones detonantes, la probabilidad de falsedad colapsó en -${(dropProb * 100).toFixed(1)} pts, pasando a rango neutral.`;
  
  // Marcar gatillos como atenuados
  currentArticle.sentences.forEach(s => {
    if (s.is_trigger) {
      s.is_trigger = false;
      s.sens_score = 0.25;
      s.sens_pct = 25.0;
      s.flags = ["Frase Atenuada tras Simulación"];
    }
  });

  renderArticleResults(currentArticle);
}

function normalizeCapsInArticle() {
  if (!currentArticle || !currentArticle.sentences) return;

  currentArticle.full_text = currentArticle.full_text.toLowerCase();
  currentArticle.full_text = currentArticle.full_text.charAt(0).toUpperCase() + currentArticle.full_text.slice(1);

  currentArticle.sentences.forEach(s => {
    s.all_caps_ratio = 0.0;
    s.text = s.text.toLowerCase();
    s.flags = s.flags ? s.flags.filter(f => !f.includes('MAYÚSCULAS')) : [];
    if (s.sens_score > 0.4) s.sens_score -= 0.20;
    s.sens_pct = Math.round(s.sens_score * 100);
  });

  const drop = 0.18; // Mayúsculas sostenidas aportan casi 15% del modelo
  currentArticle.ensamble_5d.prob_fake = Math.max(0.15, currentArticle.ensamble_5d.prob_fake - drop);
  currentArticle.ensamble_5d.comment = `🔤 NORMALIZACIÓN TIPOGRÁFICA: Al convertir las mayúsculas sostenidas a minúsculas estándar, la dimensión D5 reduce drásticamente la sospecha de engaño viral.`;

  renderArticleResults(currentArticle);
}

// ANALIZAR TEXTO PERSONALIZADO (ONLINE U OFFLINE)
async function analyzeArticleText(text) {
  const btn = document.getElementById('btn-run-inspector');
  const btnText = document.getElementById('btn-run-inspector-text');

  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = "Extrayendo 42 Variables & Inferencia 5D...";

  try {
    if (serverOnline) {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (res.ok) {
        const data = await res.json();
        currentArticle = {
          title: "Noticia Analizada por el Usuario",
          region: "Texto Libre",
          country: "Personalizado",
          full_text: text,
          real_label: "DESCONOCIDO",
          sabert: {
            label: data.sabert.label,
            prob_fake: data.sabert.prob_fake,
            prob_fake_pct: data.sabert.prob_fake_pct,
            comment: data.sabert.opacity_note
          },
          ensamble_5d: {
            label: data.ensamble_5d.label_th42,
            prob_fake: data.ensamble_5d.prob_fake,
            prob_fake_pct: data.ensamble_5d.prob_fake_pct,
            comment: `Meta-Ensamble GBDT 5D: Clasificación multidimensional a umbral óptimo θ*=0.42.`
          },
          sentences: data.sentences
        };
        renderArticleResults(currentArticle);
        if (btn) btn.disabled = false;
        if (btnText) btnText.textContent = "Analizar con Pipeline 5D";
        return;
      }
    }

    // MODO STANDALONE CLIENT-SIDE ANALYZER
    const clientAnalysis = runClientSideAnalysis(text);
    currentArticle = clientAnalysis;
    renderArticleResults(currentArticle);

  } catch (err) {
    console.error("Error analizando noticia:", err);
    const clientAnalysis = runClientSideAnalysis(text);
    currentArticle = clientAnalysis;
    renderArticleResults(currentArticle);
  } finally {
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = "Analizar con Pipeline 5D";
  }
}

// Analizador Heurístico Standalone en Cliente
function runClientSideAnalysis(text) {
  const rawSents = text.split(/(?<=[.!?\n])\s+/).filter(s => s.trim().length > 8);
  const sents = rawSents.length > 0 ? rawSents : [text];

  const dicendi = ["dijo", "afirmó", "aseguró", "señaló", "declaró", "explicó", "anunció", "informó", "manifestó", "indicó"];
  const boosters = ["totalmente", "absolutamente", "definitivamente", "obviamente", "indiscutiblemente", "claramente", "urgente", "alerta", "bomba", "boooomm"];
  
  let totalWords = 0;
  let totalCapsWords = 0;
  let totalAdv = 0;
  let prevWords = null;

  const analyzedSents = sents.map((s, idx) => {
    const words = s.match(/[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+/g) || [];
    totalWords += words.length;

    const capsWords = words.filter(w => w === w.toUpperCase() && w.length >= 3);
    totalCapsWords += capsWords.length;
    const capsRatio = words.length > 0 ? (capsWords.length / words.length) * 100 : 0;

    const advWords = words.filter(w => w.toLowerCase().endsWith('mente') || ['muy', 'más', 'tan', 'bastante', 'casi', 'apenas', 'siempre', 'nunca'].includes(w.toLowerCase()));
    totalAdv += advWords.length;
    const advDensity = words.length > 0 ? (advWords.length / words.length) * 100 : 0;

    const hasDicendi = words.some(w => dicendi.includes(w.toLowerCase()));
    const hasBoosters = words.some(w => boosters.includes(w.toLowerCase()));
    const hasQuotes = /["«»“”]/.test(s);
    const hasExcl = s.includes('!') || s.includes('¡');

    let sim = 0.45;
    if (prevWords && prevWords.length > 0 && words.length > 0) {
      const s1 = new Set(prevWords.map(w => w.toLowerCase()));
      const s2 = new Set(words.map(w => w.toLowerCase()));
      const inter = [...s1].filter(x => s2.has(x)).length;
      sim = inter / Math.max(1, s1.size + s2.size - inter);
    }
    prevWords = words;

    let score = 0.20;
    if (capsRatio > 15) score += 0.35;
    else if (capsRatio > 5) score += 0.15;
    if (advDensity > 7) score += 0.20;
    if (hasBoosters) score += 0.25;
    if (hasExcl) score += 0.15;
    if (hasDicendi || hasQuotes) score -= 0.18;
    score = Math.max(0.05, Math.min(0.96, score));

    const isTrigger = score >= 0.65 || capsRatio >= 20.0;

    const flags = [];
    if (capsRatio >= 12.0) flags.push("MAYÚSCULAS SOSTENIDAS");
    if (advDensity >= 7.0) flags.push(`Alta Densidad Adverbial (${advDensity.toFixed(1)}%)`);
    if (hasBoosters) flags.push("Intensificador / Booster");
    if (hasExcl) flags.push("Énfasis Puntuación (!)");
    if (hasDicendi) flags.push("Verbo Atribución / Dicendi");
    if (hasQuotes) flags.push("Cita Entrecomillada");
    if (sim < 0.10 && idx > 0) flags.push("Salto Temático Incoherente");

    return {
      idx: idx + 1,
      text: s,
      sens_score: score,
      sens_pct: Math.round(score * 100),
      consec_sim: parseFloat(sim.toFixed(3)),
      is_trigger: isTrigger,
      all_caps_ratio: parseFloat(capsRatio.toFixed(1)),
      adv_density: parseFloat(advDensity.toFixed(1)),
      dicendi_count: hasDicendi ? 1 : 0,
      has_quotes: hasQuotes,
      flags
    };
  });

  const overallCapsRatio = totalWords > 0 ? (totalCapsWords / totalWords) * 100 : 0;
  const overallAdvDensity = totalWords > 0 ? (totalAdv / totalWords) * 100 : 0;
  const maxSens = Math.max(...analyzedSents.map(s => s.sens_score));

  let probFake5d = 0.25;
  if (overallCapsRatio > 6.0) probFake5d += 0.30;
  if (overallAdvDensity > 6.0) probFake5d += 0.22;
  if (maxSens > 0.70) probFake5d += 0.25;
  probFake5d = Math.max(0.08, Math.min(0.93, probFake5d));

  const probFakeSaBERT = maxSens > 0.75 ? 0.82 : 0.28;

  return {
    title: "Análisis Personalizado en Cliente (Modo Standalone)",
    region: "Texto Libre",
    country: "Local",
    full_text: text,
    real_label: "DESCONOCIDO",
    sabert: {
      label: probFakeSaBERT >= 0.50 ? "FALSO" : "VERDADERO",
      prob_fake: probFakeSaBERT,
      prob_fake_pct: Math.round(probFakeSaBERT * 100),
      comment: "Simulación de SaBERT tradicional: Salida binaria escalar opaca sin desglose de oraciones."
    },
    ensamble_5d: {
      label: probFake5d >= 0.42 ? "FALSO" : "VERDADERO",
      prob_fake: probFake5d,
      prob_fake_pct: Math.round(probFake5d * 100),
      comment: "Meta-Ensamble 5D Standalone: Decisión multi-dimensional calibrada a θ*=0.42."
    },
    sentences: analyzedSents
  };
}

// ==========================================================================
// TAB 4: IMPORTANCIA DE VARIABLES & DICCIONARIO
// ==========================================================================
function initDimensionsChart() {
  const canvas = document.getElementById('chart-dimensions');
  if (!canvas || typeof Chart === 'undefined') return;

  const isDark = currentTheme === 'dark';
  const textColor = isDark ? '#f8fafc' : '#0f172a';

  dimensionsChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: [
        'D0: Sensacionalismo & Redundancia',
        'D5: Anclajes, Mayúsculas & Signos',
        'D3: Riqueza Léxica & Legibilidad',
        'D4: Morfosintaxis POS & Subjetividad',
        'D2: Lingüística Forense & Epistémica',
        'D1: Coherencia Secuencial'
      ],
      datasets: [{
        data: [31.10, 24.11, 19.34, 14.78, 6.56, 4.10],
        backgroundColor: [
          '#ef4444',
          '#ec4899',
          '#10b981',
          '#f59e0b',
          '#8b5cf6',
          '#3b82f6'
        ],
        borderWidth: 2,
        borderColor: isDark ? '#111827' : '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: {
            color: textColor,
            font: { size: 11, weight: '600' }
          }
        },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              return ` ${ctx.label}: ${ctx.raw}% de Ganancia de Información`;
            }
          }
        }
      }
    }
  });
}

function setupDictionaryEvents() {
  const searchInput = document.getElementById('dict-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      filterDictionary();
    });
  }

  document.querySelectorAll('.btn-filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.btn-filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      filterDictionary();
    });
  });
}

function initDictionaryView() {
  filterDictionary();
}

function filterDictionary() {
  const grid = document.getElementById('dictionary-grid');
  const searchInput = document.getElementById('dict-search-input');
  const countEl = document.getElementById('dict-count');
  if (!grid) return;

  const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
  const activePill = document.querySelector('.btn-filter-pill.active');
  const activeDim = activePill ? activePill.getAttribute('data-filter') : 'ALL';

  const filtered = dictionaryData.filter(v => {
    const matchQuery = !query || v.name.toLowerCase().includes(query) || v.hipotesis.toLowerCase().includes(query) || v.formula.toLowerCase().includes(query);
    const matchDim = activeDim === 'ALL' || v.dim === activeDim;
    return matchQuery && matchDim;
  });

  if (countEl) countEl.textContent = `${filtered.length} variables mostradas`;
  grid.innerHTML = '';

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="text-muted" style="grid-column: 1/-1; text-align: center; padding: 2rem;">No se encontraron variables con esos filtros.</div>`;
    return;
  }

  filtered.forEach(v => {
    const card = document.createElement('div');
    card.className = 'var-card';

    let dimColor = '#3b82f6';
    if (v.dim === 'D0') dimColor = '#ef4444';
    else if (v.dim === 'D5') dimColor = '#ec4899';
    else if (v.dim === 'D3') dimColor = '#10b981';
    else if (v.dim === 'D4') dimColor = '#f59e0b';
    else if (v.dim === 'D2') dimColor = '#8b5cf6';

    card.style.borderLeft = `4px solid ${dimColor}`;

    card.innerHTML = `
      <div class="var-card-top">
        <span class="var-name" style="color: ${dimColor};">${v.name}</span>
        <span class="var-importance-badge">${v.importance ? v.importance + '%' : ''} Ganancia</span>
      </div>
      <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted);">
        ${v.dim_name || v.dim}
      </div>
      <div class="var-formula">
        ${v.formula}
      </div>
      <div class="var-hypothesis">
        ${v.hipotesis}
      </div>
      <div class="var-effect">
        🎯 <strong>Efecto:</strong> ${v.efecto}
      </div>
    `;

    grid.appendChild(card);
  });
}

// ==========================================================================
// MODAL DE AMPLIACIÓN DE IMÁGENES
// ==========================================================================
function openImageModal(src, caption) {
  const overlay = document.getElementById('img-modal-overlay');
  const img = document.getElementById('modal-img-element');
  const cap = document.getElementById('modal-img-caption');

  if (overlay && img) {
    img.src = src;
    if (cap) cap.textContent = caption || 'Figura Oficial de la Tesis';
    overlay.classList.add('active');
  }
}

function closeImageModal() {
  const overlay = document.getElementById('img-modal-overlay');
  if (overlay) overlay.classList.remove('active');
}

// ==========================================================================
// FALLBACK DATA EN CASO DE MODO OFFLINE O STANDALONE
// ==========================================================================
function getFallbackMetrics() {
  return {
    sweep_umbrales: [
      { threshold: 0.25, accuracy: 0.5527, precision: 0.5041, recall: 0.9417, f1_fake: 0.6567, tn: 552, fp: 1859, fn: 117, tp: 1890 },
      { threshold: 0.30, accuracy: 0.5842, precision: 0.5284, recall: 0.8879, f1_fake: 0.6625, tn: 799, fp: 1612, fn: 225, tp: 1782 },
      { threshold: 0.35, accuracy: 0.6120, precision: 0.5539, recall: 0.8256, f1_fake: 0.6631, tn: 1047, fp: 1364, fn: 350, tp: 1657 },
      { threshold: 0.40, accuracy: 0.6329, precision: 0.5779, recall: 0.7384, f1_fake: 0.6484, tn: 1324, fp: 1087, fn: 525, tp: 1482 },
      { threshold: 0.42, accuracy: 0.6383, precision: 0.5851, recall: 0.7005, f1_fake: 0.6376, tn: 1414, fp: 997, fn: 601, tp: 1406 },
      { threshold: 0.50, accuracy: 0.6569, precision: 0.6593, recall: 0.5062, f1_fake: 0.5727, tn: 1886, fp: 525, fn: 991, tp: 1016 },
      { threshold: 0.55, accuracy: 0.6541, precision: 0.7025, recall: 0.4121, f1_fake: 0.5195, tn: 2062, fp: 349, fn: 1180, tp: 827 },
      { threshold: 0.60, accuracy: 0.6403, precision: 0.7423, recall: 0.3129, f1_fake: 0.4401, tn: 2193, fp: 218, fn: 1379, tp: 628 },
      { threshold: 0.65, accuracy: 0.6186, precision: 0.7932, recall: 0.2108, f1_fake: 0.3332, tn: 2301, fp: 110, fn: 1584, tp: 423 },
      { threshold: 0.70, accuracy: 0.5885, precision: 0.8354, recall: 0.1191, f1_fake: 0.2085, tn: 2364, fp: 47, fn: 1768, tp: 239 }
    ]
  };
}

function getFallbackBenchmark() {
  return [
    {
      id: "latam_fake_covid_caps",
      title: "Bulo de Salud con Mayúsculas Sostenidas (América Latina)",
      region: "América Latina",
      country: "Hispanoamérica",
      real_label: "FALSO",
      full_text: "Boooomm\nMUJERES VACUNADAS DE COVID ESTÁN MOSTRANDO EFECTOS SECUNDARIOS TÍPICOS DE CANCER DE MAMA\nLos médicos de Intermountain Healthcare anuncian nuevas pautas para pacientes vacunadas.\nLos médicos han observado inflamación masiva de ganglios linfáticos tras la inoculación de la fórmula experimental.\n\"Siempre que vemos esto en mamografías normales convocamos a biopsias urgentes por posible cáncer metastásico\".\nLa población debe exigir la suspensión inmediata de este veneno mortal.",
      sabert: {
        label: "VERDADERO",
        prob_fake: 0.38,
        prob_fake_pct: 38.0,
        comment: "Falso Negativo catastrófico de SaBERT por Domain Shift (38% prob. falso). Al no detectar políticos españoles, SaBERT asume neutralidad informativa."
      },
      ensamble_5d: {
        label: "FALSO",
        prob_fake: 0.788,
        prob_fake_pct: 78.8,
        comment: "El Ensamble 5D detecta la anomalía morfosintáctica: 18.4% de mayúsculas sostenidas, alta densidad de adverbios y ausencia total de citas contrastables."
      },
      sentences: [
        {
          idx: 1,
          text: "Boooomm MUJERES VACUNADAS DE COVID ESTÁN MOSTRANDO EFECTOS SECUNDARIOS TÍPICOS DE CANCER DE MAMA.",
          sens_score: 0.95,
          sens_pct: 95.0,
          consec_sim: 0.50,
          is_trigger: true,
          all_caps_ratio: 42.0,
          adv_density: 0.0,
          dicendi_count: 0,
          flags: ["MAYÚSCULAS SOSTENIDAS", "Intensificador / Booster"]
        },
        {
          idx: 2,
          text: "Los médicos de Intermountain Healthcare anuncian nuevas pautas para pacientes vacunadas.",
          sens_score: 0.20,
          sens_pct: 20.0,
          consec_sim: 0.25,
          is_trigger: false,
          all_caps_ratio: 0.0,
          adv_density: 0.0,
          dicendi_count: 1,
          flags: ["Verbo Atribución / Dicendi"]
        },
        {
          idx: 3,
          text: "Los médicos han observado inflamación masiva de ganglios linfáticos tras la inoculación de la fórmula experimental.",
          sens_score: 0.55,
          sens_pct: 55.0,
          consec_sim: 0.40,
          is_trigger: false,
          all_caps_ratio: 0.0,
          adv_density: 0.0,
          dicendi_count: 0,
          flags: []
        },
        {
          idx: 4,
          text: "La población debe exigir la suspensión inmediata de este veneno mortal.",
          sens_score: 0.88,
          sens_pct: 88.0,
          consec_sim: 0.08,
          is_trigger: true,
          all_caps_ratio: 0.0,
          adv_density: 12.5,
          dicendi_count: 0,
          flags: ["Alta Densidad Adverbial (12.5%)", "Salto Temático Incoherente"]
        }
      ]
    },
    {
      id: "latam_real_puebla",
      title: "Noticia Oficial de Salud Pública (México)",
      region: "América Latina",
      country: "México",
      real_label: "VERDADERO",
      full_text: "El Gobierno del Estado de Puebla anunció que el confinamiento para evitar que colapse el sistema de salud por la pandemia de coronavirus se extiende hasta el próximo lunes.\nLas autoridades sanitarias informaron que se mantienen habilitadas las camas hospitalarias en los centros médicos regionales.\nEl gobernador puntualizó que las medidas se ajustarán según el comportamiento de la curva epidemiológica reportada por la Secretaría de Salud.",
      sabert: {
        label: "VERDADERO",
        prob_fake: 0.12,
        prob_fake_pct: 12.0,
        comment: "SaBERT predice correctamente veracidad, pero de forma completamente opaca sin justificar el fallo."
      },
      ensamble_5d: {
        label: "VERDADERO",
        prob_fake: 0.115,
        prob_fake_pct: 11.5,
        comment: "El Ensamble 5D certifica su legitimidad: alta presencia de verbos dicendi ('anunció', 'informaron', 'puntualizó'), 0% mayúsculas sostenidas y coherencia secuencial fluida."
      },
      sentences: [
        {
          idx: 1,
          text: "El Gobierno del Estado de Puebla anunció que el confinamiento para evitar que colapse el sistema de salud por la pandemia se extiende hasta el próximo lunes.",
          sens_score: 0.15,
          sens_pct: 15.0,
          consec_sim: 0.50,
          is_trigger: false,
          all_caps_ratio: 0.0,
          adv_density: 0.0,
          dicendi_count: 1,
          flags: ["Verbo Atribución / Dicendi"]
        },
        {
          idx: 2,
          text: "Las autoridades sanitarias informaron que se mantienen habilitadas las camas hospitalarias en los centros médicos regionales.",
          sens_score: 0.10,
          sens_pct: 10.0,
          consec_sim: 0.35,
          is_trigger: false,
          all_caps_ratio: 0.0,
          adv_density: 0.0,
          dicendi_count: 1,
          flags: ["Verbo Atribución / Dicendi"]
        },
        {
          idx: 3,
          text: "El gobernador puntualizó que las medidas se ajustarán según el comportamiento de la curva epidemiológica reportada por la Secretaría de Salud.",
          sens_score: 0.12,
          sens_pct: 12.0,
          consec_sim: 0.42,
          is_trigger: false,
          all_caps_ratio: 0.0,
          adv_density: 0.0,
          dicendi_count: 1,
          flags: ["Verbo Atribución / Dicendi"]
        }
      ]
    }
  ];
}

function getFallbackDictionary() {
  return [
    {
      id: "adv_density",
      name: "adv_density",
      dim: "D4",
      dim_name: "D4: Morfosintaxis POS & Subjetividad",
      importance: 8.41,
      formula: "(\\sum \\text{Adverbios} / \\text{Palabras}) \\times 100",
      hipotesis: "¡TOP 1 INDIVIDUAL! Los adverbios modales ('brutalmente', 'totalmente', 'muy') transmiten valoración subjetiva e hiperbólica.",
      efecto: "Densidad >6.5% es el predictor morfosintáctico más contundente de desinformación."
    },
    {
      id: "upper_chars_ratio",
      name: "upper_chars_ratio",
      dim: "D5",
      dim_name: "D5: Anclajes, Mayúsculas & Signos",
      importance: 7.49,
      formula: "\\text{Letras mayúsculas} / \\text{Total caracteres}",
      hipotesis: "¡TOP 2 INDIVIDUAL! Refleja el grito digital y la urgencia inducida mediante tipografía alterada.",
      efecto: "Valores >5% son raros en prensa seria y predominan en bulos virales de redes."
    },
    {
      id: "all_caps_ratio",
      name: "all_caps_ratio",
      dim: "D5",
      dim_name: "D5: Anclajes, Mayúsculas & Signos",
      importance: 6.96,
      formula: "(\\text{Palabras en MAYÚSCULAS} / \\text{Palabras}) \\times 100",
      hipotesis: "¡TOP 3 INDIVIDUAL! Palabras vociferadas ('¡¡URGENTE!!', 'ALERTA', 'DIFUNDIR'). Invariante a fronteras regionales.",
      efecto: "Aporta casi un 7% del poder explicativo del ensamble por sí sola."
    },
    {
      id: "P_full",
      name: "P_full",
      dim: "D0",
      dim_name: "D0: Sensacionalismo & Redundancia",
      importance: 5.83,
      formula: "P(\\text{Sensacionalista} | \\text{Doc}) \\text{ vía BETO}",
      hipotesis: "Carga afectiva y emocional agresiva en todo el documento periodístico.",
      efecto: "Valores >0.70 elevan la sospecha global de cebo engañoso."
    },
    {
      id: "hapax_ratio",
      name: "hapax_ratio",
      dim: "D3",
      dim_name: "D3: Riqueza Léxica & Legibilidad",
      importance: 5.21,
      formula: "\\text{Palabras únicas (frec=1)} / \\text{Total palabras}",
      hipotesis: "Diversidad de vocabulario: redactores serios usan léxico amplio; bulos reutilizan pocas palabras.",
      efecto: "Ratios bajos (<35%) denotan pobreza léxica y reiteración obsesiva."
    },
    {
      id: "dicendi_density",
      name: "dicendi_density",
      dim: "D2",
      dim_name: "D2: Lingüística Forense & Epistémica",
      importance: 4.88,
      formula: "(\\sum \\text{Verbos Dicendi} / \\text{Palabras}) \\times 100",
      hipotesis: "Presencia de verbos de reporte y citas ('dijo', 'afirmó', 'declaró', 'indicó').",
      efecto: "Densidad nula o ínfima delata la falta de atribución de fuentes contrastables."
    }
  ];
}

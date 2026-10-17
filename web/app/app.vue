<template>
  <div class="page">
    <!-- NAV -->
    <header class="nav">
      <div class="nav-inner">
        <div class="brand"><span class="brand-mark">◈</span> FACE&nbsp;SMASH</div>
        <nav class="nav-links">
          <a href="#studio">Studio</a>
          <a href="#how">How it works</a>
          <a href="#limits">Limits</a>
        </nav>
        <div class="cup-badge">AI BUILDERS CUP 2026</div>
      </div>
    </header>

    <!-- HERO -->
    <section class="hero">
      <p class="kicker">Anti facial-recognition &nbsp;·&nbsp; PGD cloaking</p>
      <h1>Smash the algorithms.<br><span class="accent">Keep your face.</span></h1>
      <p class="lede">
        Every photo you post trains someone else's AI. Face Smash cloaks your photos with
        invisible adversarial noise — your friends see you, face-search engines don't.
      </p>
      <div class="hero-cta">
        <a href="#studio" class="btn btn-primary">Cloak a photo</a>
        <a href="#how" class="btn btn-ghost">How it works</a>
      </div>
      <div class="stats">
        <div class="stat"><strong>1.52</strong><span>embedding distance<br>(identity threshold ~1.0)</span></div>
        <div class="stat"><strong>32.7<span class="unit">dB</span></strong><span>PSNR — invisible<br>to the human eye</span></div>
        <div class="stat"><strong>Q75 ✓</strong><span>survives JPEG<br>recompression</span></div>
      </div>
    </section>

    <!-- STUDIO -->
    <section id="studio" class="studio">
      <h2>The Studio</h2>
      <p class="section-sub">Upload a photo. We break its AI identity — you keep the pixels.</p>

      <div class="studio-grid">
        <!-- INPUT -->
        <div class="card">
          <h3>1 · Upload</h3>
          <div
            class="dropzone" :class="{ dragging }"
            @dragover.prevent="dragging = true" @dragleave="dragging = false"
            @drop.prevent="onDrop" @click="fileInput?.click()"
          >
            <template v-if="!previewUrl">
              <div class="dz-icon">⇪</div>
              <p><strong>Drop a photo here</strong> or click to browse</p>
              <p class="muted small">JPG / PNG / WebP · front-facing face works best</p>
            </template>
            <template v-else>
              <img :src="previewUrl" alt="preview" class="dz-preview">
              <p class="muted small">Click to replace</p>
            </template>
            <input ref="fileInput" type="file" accept="image/jpeg,image/png,image/webp" hidden @change="onFile">
          </div>

          <h3>2 · Strength</h3>
          <div class="seg">
            <button v-for="s in strengths" :key="s.id"
              :class="['seg-btn', { active: strength === s.id }]"
              @click="strength = s.id">
              {{ s.label }}<span class="seg-sub">{{ s.time }}</span>
            </button>
          </div>

          <button class="btn btn-primary btn-block" :disabled="!previewUrl || loading" @click="cloak">
            <span v-if="!loading">⚡ Cloak my photo</span>
            <span v-else class="spinner"></span>
          </button>
          <p v-if="loading" class="muted small center pulse">{{ loadingMsg }}</p>
          <p v-if="error" class="error">⚠ {{ error }}</p>
        </div>

        <!-- RESULT -->
        <div class="card">
          <h3>3 · Result</h3>
          <div v-if="!result" class="empty-result">
            <div class="empty-icon">◌</div>
            <p class="muted">Your cloaked photo will appear here.<br>Same face to humans — a stranger to AI.</p>
          </div>
          <template v-else>
            <div class="compare" ref="compareBox">
              <img :src="resultUrl" alt="cloaked" class="cmp-img">
              <div class="cmp-top" :style="{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }">
                <img :src="previewUrl" alt="original" class="cmp-img">
              </div>
              <span class="cmp-tag left">original</span>
              <span class="cmp-tag right">cloaked</span>
              <input type="range" min="0" max="100" v-model.number="sliderPos" class="cmp-slider" aria-label="Compare slider">
            </div>

            <div class="verdict" :class="verdictClass">
              {{ verdictText }}
            </div>

            <div class="metrics">
              <div class="metric">
                <span class="m-label">Embedding distance</span>
                <strong class="m-val">{{ result.metrics.embedding_distance }}</strong>
                <div class="bar"><div class="bar-fill" :style="{ width: distPct + '%' }"></div><div class="bar-threshold"></div></div>
                <span class="m-note">threshold ~1.0 · farther = safer</span>
              </div>
              <div class="metric"><span class="m-label">Visual quality (PSNR)</span><strong class="m-val">{{ result.metrics.psnr_db }} dB</strong><span class="m-note">above 30 dB ≈ invisible</span></div>
              <div class="metric"><span class="m-label">Cosine similarity</span><strong class="m-val">{{ result.metrics.cosine_similarity }}</strong><span class="m-note">lower = less recognizable</span></div>
              <div class="metric"><span class="m-label">JPEG Q75 upload</span><strong class="m-val" :class="result.metrics.jpeg_q75_survived ? 'good' : 'bad'">{{ result.metrics.jpeg_q75_survived ? '✓ survives' : '✗ fails' }}</strong><span class="m-note">simulated social-media compression</span></div>
              <div class="metric"><span class="m-label">Face detection</span><strong class="m-val">{{ result.metrics.detection_before }} → {{ result.metrics.detection_after }}</strong><span class="m-note">still a face — just not yours</span></div>
              <div class="metric"><span class="m-label">Processing time</span><strong class="m-val">{{ (result.meta.processing_time_ms / 1000).toFixed(1) }}s</strong><span class="m-note">{{ result.meta.strength }} · seed {{ result.meta.seed }}</span></div>
            </div>

            <div class="result-actions">
              <button class="btn btn-primary" @click="download">⬇ Download cloaked PNG</button>
              <button class="btn btn-ghost" :disabled="reportLoading" @click="explain">
                {{ reportLoading ? '…' : '✨ Explain with AI' }}
              </button>
            </div>
            <div v-if="report" class="report"><p class="report-label">AI report</p><p>{{ report }}</p></div>
          </template>
        </div>
      </div>
    </section>

    <!-- HOW -->
    <section id="how" class="how">
      <h2>How it works</h2>
      <div class="how-grid">
        <div class="card"><div class="how-n">01</div><h3>Detect</h3><p>MTCNN finds your face and crops a 160×160 region for the cloak.</p></div>
        <div class="card"><div class="how-n">02</div><h3>Perturb</h3><p>60 steps of Projected Gradient Descent (ε 8/255) push the face embedding far from your identity — against an InceptionResnetV1 surrogate.</p></div>
        <div class="card"><div class="how-n">03</div><h3>Blend back</h3><p>The cloaked face is pasted back into your original photo. PSNR 32.7 dB — you can't see the difference, AI can't see you.</p></div>
      </div>
    </section>

    <!-- LIMITS -->
    <section id="limits" class="limits">
      <h2>Honest limits</h2>
      <ul>
        <li><strong>Breaks identity matching, not face detection.</strong> A detector still sees a face — a matcher can't tell whose it is.</li>
        <li><strong>Validated against one surrogate model</strong> (InceptionResnetV1 / VGGFace2) in simulation — not a claim to defeat commercial black-box systems.</li>
        <li><strong>Strong protection, not a guarantee</strong> against all current and future AI.</li>
      </ul>
    </section>

    <footer>
      <p><span class="brand-mark">◈</span> <strong>FACE SMASH</strong> · team HelloWorld</p>
      <div class="chips">
        <span>Cloud Run</span><span>Firestore</span><span>Gemini</span><span>Firebase</span><span>Nuxt</span>
      </div>
      <p class="muted small">Google Cloud AI Builders Cup 2026 · JAPAC</p>
    </footer>
  </div>
</template>

<script setup>
const fileInput = ref(null)
const previewUrl = ref('')
const previewFile = ref(null)
const dragging = ref(false)
const strength = ref('standard')
const loading = ref(false)
const loadingMsg = ref('')
const result = ref(null)
const error = ref('')
const sliderPos = ref(50)
const report = ref('')
const reportLoading = ref(false)

const strengths = [
  { id: 'low', label: 'Low', time: '~30s' },
  { id: 'standard', label: 'Standard', time: '~60s' },
  { id: 'max', label: 'Max', time: '~2min' },
]

const resultUrl = computed(() =>
  result.value ? `data:image/png;base64,${result.value.protected_image}` : '')

const verdictText = computed(() => {
  if (!result.value) return ''
  const d = result.value.metrics.embedding_distance
  if (d >= 1.2) return '✓ IDENTITY MATCH BROKEN'
  if (d >= 1.0) return '◐ BORDERLINE — try Max strength'
  return '✗ WEAK — try Max strength'
})
const verdictClass = computed(() => {
  if (!result.value) return ''
  const d = result.value.metrics.embedding_distance
  return d >= 1.2 ? 'good' : d >= 1.0 ? 'warn' : 'bad'
})
const distPct = computed(() =>
  result.value ? Math.min(100, (result.value.metrics.embedding_distance / 2) * 100) : 0)

function setFile(f) {
  if (!f || !/^image\/(jpeg|png|webp)$/.test(f.type)) {
    error.value = "That file type isn't supported — use JPG, PNG or WebP."
    return
  }
  if (f.size > 10 * 1024 * 1024) {
    error.value = 'Image is larger than 10 MB — please downscale it first.'
    return
  }
  error.value = ''
  result.value = null
  report.value = ''
  previewFile.value = f
  previewUrl.value = URL.createObjectURL(f)
}
const onFile = (e) => setFile(e.target.files?.[0])
const onDrop = (e) => { dragging.value = false; setFile(e.dataTransfer.files?.[0]) }

const loadingSteps = ['Uploading…', 'Detecting face…', 'Running PGD cloak…', 'Verifying metrics…']

async function cloak() {
  if (!previewFile.value || loading.value) return
  loading.value = true
  error.value = ''
  result.value = null
  report.value = ''
  let i = 0
  loadingMsg.value = loadingSteps[0]
  const timer = setInterval(() => { i = (i + 1) % loadingSteps.length; loadingMsg.value = loadingSteps[i] }, 9000)
  try {
    const fd = new FormData()
    fd.append('image', previewFile.value)
    fd.append('strength', strength.value)
    const res = await $fetch('/api/protect', { method: 'POST', body: fd, timeout: 290000 })
    result.value = res
    sliderPos.value = 50
  }
  catch (e) {
    error.value = e?.data?.message || e?.message || 'Something went wrong.'
  }
  finally {
    clearInterval(timer)
    loading.value = false
  }
}

async function explain() {
  if (!result.value || reportLoading.value) return
  reportLoading.value = true
  try {
    const res = await $fetch('/api/report', {
      method: 'POST',
      body: { metrics: result.value.metrics, meta: result.value.meta, lang: 'en' },
    })
    report.value = res.report + (res.model === 'mock' ? '  [offline summary]' : '')
  }
  catch {
    report.value = 'Could not generate the report right now.'
  }
  finally {
    reportLoading.value = false
  }
}

function download() {
  const a = document.createElement('a')
  a.href = resultUrl.value
  a.download = 'facesmash-cloaked.png'
  a.click()
}
</script>

<style>
:root {
  --bg: #07090d; --surface: #0d1218; --surface2: #131a23;
  --border: #1f2833; --text: #e8eef4; --muted: #8b98a5;
  --accent: #b4f34a; --accent-ink: #0c1206;
  --danger: #ff6b6b; --warn: #ffd166; --cyan: #67e8f9;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--text);
  font-family: "Space Grotesk", system-ui, -apple-system, sans-serif; }
.page { min-height: 100vh; }
.muted { color: var(--muted); } .small { font-size: .85rem; }
.center { text-align: center; }
.accent { color: var(--accent); }

/* NAV */
.nav { position: sticky; top: 0; z-index: 50; background: rgba(7,9,13,.85);
  backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); }
.nav-inner { max-width: 1080px; margin: 0 auto; padding: .9rem 1.25rem;
  display: flex; align-items: center; gap: 1.5rem; }
.brand { font-weight: 800; letter-spacing: .12em; font-size: .95rem; }
.brand-mark { color: var(--accent); }
.nav-links { display: flex; gap: 1.25rem; margin-left: auto; }
.nav-links a { color: var(--muted); text-decoration: none; font-size: .9rem; }
.nav-links a:hover { color: var(--text); }
.cup-badge { font-size: .68rem; letter-spacing: .14em; color: var(--accent);
  border: 1px solid var(--accent); border-radius: 999px; padding: .3rem .7rem; white-space: nowrap; }

/* HERO */
.hero { max-width: 1080px; margin: 0 auto; padding: 5rem 1.25rem 3rem; text-align: center; }
.kicker { color: var(--accent); letter-spacing: .22em; font-size: .75rem; text-transform: uppercase; }
.hero h1 { font-size: clamp(2.4rem, 6vw, 4.2rem); line-height: 1.05; margin: 1rem 0; letter-spacing: -0.02em; }
.lede { color: var(--muted); font-size: 1.1rem; max-width: 620px; margin: 0 auto 2rem; line-height: 1.6; }
.hero-cta { display: flex; gap: .75rem; justify-content: center; margin-bottom: 3rem; }
.stats { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 1.1rem 1.6rem; min-width: 170px; }
.stat strong { display: block; font-size: 1.7rem; color: var(--accent); font-family: "IBM Plex Mono", ui-monospace, monospace; }
.stat .unit { font-size: 1rem; } .stat span:last-child { color: var(--muted); font-size: .8rem; line-height: 1.4; display: block; margin-top: .3rem; }

/* BUTTONS */
.btn { display: inline-flex; align-items: center; justify-content: center; gap: .5rem;
  padding: .8rem 1.6rem; border-radius: 10px; font-weight: 700; font-size: .95rem;
  cursor: pointer; text-decoration: none; border: 1px solid transparent; transition: transform .12s; }
.btn:active { transform: scale(.98); }
.btn-primary { background: var(--accent); color: var(--accent-ink); }
.btn-primary:hover { filter: brightness(1.08); }
.btn-primary:disabled { opacity: .45; cursor: not-allowed; }
.btn-ghost { background: transparent; color: var(--text); border-color: var(--border); }
.btn-ghost:hover { border-color: var(--muted); }
.btn-ghost:disabled { opacity: .45; cursor: not-allowed; }
.btn-block { width: 100%; margin-top: 1rem; }

/* SECTIONS */
.studio, .how, .limits { max-width: 1080px; margin: 0 auto; padding: 3rem 1.25rem; }
.studio h2, .how h2, .limits h2 { font-size: 1.9rem; margin-bottom: .4rem; }
.section-sub { color: var(--muted); margin-bottom: 1.75rem; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; }
.card h3 { margin: 0 0 1rem; font-size: 1rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-size: .8rem; }

/* STUDIO */
.studio-grid { display: grid; grid-template-columns: 1fr 1.35fr; gap: 1.25rem; }
@media (max-width: 860px) { .studio-grid { grid-template-columns: 1fr; } }
.dropzone { border: 2px dashed var(--border); border-radius: 12px; padding: 2rem 1rem;
  text-align: center; cursor: pointer; transition: border-color .15s, background .15s; margin-bottom: 1.5rem; }
.dropzone:hover, .dropzone.dragging { border-color: var(--accent); background: rgba(180,243,74,.04); }
.dz-icon { font-size: 2rem; color: var(--accent); }
.dz-preview { max-width: 100%; max-height: 220px; border-radius: 8px; }
.seg { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; }
.seg-btn { background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  border-radius: 10px; padding: .7rem .4rem; cursor: pointer; font-weight: 700; font-size: .9rem;
  display: flex; flex-direction: column; gap: .15rem; }
.seg-btn .seg-sub { font-weight: 400; font-size: .72rem; color: var(--muted); }
.seg-btn.active { border-color: var(--accent); background: rgba(180,243,74,.08); }
.error { color: var(--danger); background: rgba(255,107,107,.08); border: 1px solid rgba(255,107,107,.3);
  border-radius: 10px; padding: .7rem .9rem; font-size: .88rem; margin-top: 1rem; }
.spinner { width: 18px; height: 18px; border: 2px solid rgba(12,18,6,.3); border-top-color: var(--accent-ink);
  border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }
.pulse { animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .5; } }

.empty-result { text-align: center; padding: 3.5rem 1rem; color: var(--muted); }
.empty-icon { font-size: 2.4rem; margin-bottom: .75rem; opacity: .5; }

/* COMPARE */
.compare { position: relative; border-radius: 12px; overflow: hidden; border: 1px solid var(--border); user-select: none; }
.cmp-img { display: block; width: 100%; height: auto; max-height: 420px; object-fit: contain; background: #000; }
.cmp-top { position: absolute; inset: 0; }
.cmp-top .cmp-img { height: 100%; object-fit: cover; }
.cmp-tag { position: absolute; top: .6rem; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
  background: rgba(0,0,0,.65); padding: .25rem .6rem; border-radius: 999px; }
.cmp-tag.left { left: .6rem; } .cmp-tag.right { right: .6rem; color: var(--accent); }
.cmp-slider { position: absolute; inset: 0; width: 100%; opacity: 0; cursor: ew-resize; margin: 0; }

.verdict { margin: 1rem 0; padding: .8rem; border-radius: 10px; font-weight: 800; letter-spacing: .06em;
  text-align: center; font-size: .95rem; }
.verdict.good { background: rgba(180,243,74,.1); color: var(--accent); border: 1px solid rgba(180,243,74,.35); }
.verdict.warn { background: rgba(255,209,102,.1); color: var(--warn); border: 1px solid rgba(255,209,102,.35); }
.verdict.bad { background: rgba(255,107,107,.1); color: var(--danger); border: 1px solid rgba(255,107,107,.35); }

.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin-bottom: 1.25rem; }
@media (max-width: 640px) { .metrics { grid-template-columns: repeat(2, 1fr); } }
.metric { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: .8rem; }
.m-label { display: block; font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: .3rem; }
.m-val { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 1.05rem; }
.m-val.good { color: var(--accent); } .m-val.bad { color: var(--danger); }
.m-note { display: block; font-size: .72rem; color: var(--muted); margin-top: .25rem; }
.bar { position: relative; height: 6px; background: #0a0e13; border-radius: 999px; margin: .5rem 0 .3rem; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 999px; transition: width .8s ease; }
.bar-threshold { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; background: var(--warn); }

.result-actions { display: flex; gap: .75rem; flex-wrap: wrap; }
.report { margin-top: 1rem; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 1rem 1.1rem; }
.report-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: var(--accent); margin: 0 0 .5rem; }
.report p:last-child { margin: 0; line-height: 1.65; font-size: .93rem; }

/* HOW / LIMITS */
.how-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
@media (max-width: 860px) { .how-grid { grid-template-columns: 1fr; } }
.how-n { font-family: "IBM Plex Mono", ui-monospace, monospace; color: var(--accent); font-size: .85rem; margin-bottom: .5rem; }
.how .card h3, .limits h2 { text-transform: none; letter-spacing: 0; font-size: 1.05rem; color: var(--text); }
.how .card p { color: var(--muted); font-size: .92rem; line-height: 1.6; margin: 0; }
.limits ul { list-style: none; padding: 0; display: grid; gap: .75rem; }
.limits li { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 1rem 1.25rem; font-size: .93rem; line-height: 1.6; color: var(--muted); }
.limits li strong { color: var(--text); }

/* FOOTER */
footer { border-top: 1px solid var(--border); margin-top: 3rem; padding: 2.5rem 1.25rem; text-align: center; }
.chips { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0; }
.chips span { font-size: .75rem; color: var(--muted); border: 1px solid var(--border);
  border-radius: 999px; padding: .3rem .8rem; }
</style>

"use strict";

const TARGET_RATE = 16000; // whisper expects 16 kHz mono

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const talkBtn = document.getElementById("talk");
const talkLabelEl = talkBtn.querySelector(".talk-label");
const hintEl = document.getElementById("hint");
const ttsEngineEl = document.getElementById("ttsEngine");
const sttLangEl = document.getElementById("sttLang");
const respLangEl = document.getElementById("respLang");
const voiceEl = document.getElementById("voice");
const gpuTextEl = document.getElementById("gpuText");
const gpuBarEl = document.getElementById("gpuBar");
const ctxTextEl = document.getElementById("ctxText");
const ctxBarEl = document.getElementById("ctxBar");
const modeRadios = document.querySelectorAll('input[name="mode"]');
const vadPanelEl = document.getElementById("vadPanel");
const threshEl = document.getElementById("thresh");
const threshTextEl = document.getElementById("threshText");
const meterFillEl = document.getElementById("meterFill");
const meterThreshEl = document.getElementById("meterThresh");
const sessionListEl = document.getElementById("sessionList");
const newSessionBtn = document.getElementById("newSession");
const sidebarEl = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");

let ws = null;
let ttsSampleRate = 24000;
let languages = [];       // active engine's reply languages: [{key, name, voices:[...]}]
let sttLanguages = [];    // engine-agnostic STT languages: [{key, name}]
let engines = [];         // [{id, name}]
let activeEngine = null;  // id of the currently deployed TTS engine
let uiLocked = false;     // true while a TTS engine is being (re)deployed
let activeSessionId = null; // id of the session shown in the log (null = unsaved/new)

// Per-turn rendering state
let currentBot = null;   // {div, content} for the in-progress assistant bubble
let currentBotRaw = "";  // accumulated raw markdown for that bubble
let currentTool = null;  // the active "searching" blob element

// --- Capture state ---
let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let captureBuffers = [];
let captureRate = 48000;
let recording = false; // true while frames are being accumulated into captureBuffers

// --- Input mode ---
// "push": hold the button / spacebar to talk.
// "vad":  hands-free; auto-capture whenever the mic level passes the threshold.
let inputMode = "push";

// --- Threshold (VAD) state ---
const VAD = {
  PREROLL_MS: 300,   // audio kept before the trigger so word onsets aren't clipped
  HANGOVER_MS: 700,  // trailing silence that ends an utterance
  MIN_SPEECH_MS: 250, // shortest segment we bother sending
  SMOOTH: 0.85,      // level meter smoothing (0..1, higher = smoother)
};
let vadThreshold = Number(threshEl.value); // dBFS
let vadEnabled = false;   // hands-free loop is on
let vadActive = false;    // currently inside a detected speech segment
let vadLevelDb = -100;    // smoothed mic level for the meter
let vadLastVoice = 0;     // performance.now() of the last above-threshold frame
let vadSegStart = 0;      // performance.now() when the current segment began
let prerollFrames = [];   // recent frames kept for pre-roll
let turnBusy = false;     // a reply is being generated/played; pause VAD to avoid echo

// --- Playback state ---
let playCtx = null;
let nextPlayTime = 0;

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

function scrollLog() {
  // Defer to the next frame so the just-inserted/updated content is laid out
  // before we measure scrollHeight; otherwise we scroll to the stale height and
  // stop short of the real bottom.
  requestAnimationFrame(() => {
    logEl.scrollTop = logEl.scrollHeight;
  });
}

const ICON_PLAY =
  '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M4 3l9 5-9 5z" fill="currentColor"/></svg>';
const ICON_PAUSE =
  '<svg viewBox="0 0 16 16" width="13" height="13">' +
  '<rect x="4" y="3" width="3.4" height="10" rx="1" fill="currentColor"/>' +
  '<rect x="8.6" y="3" width="3.4" height="10" rx="1" fill="currentColor"/></svg>';

// At most one replay clip plays at a time: {audio, btn} of the active one.
let activeReplay = null;

function makeMsg(kind, who, replay) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  if (who) {
    const label = document.createElement("span");
    label.className = "who";
    label.textContent = who;
    div.appendChild(label);
  }
  const content = document.createElement("div");
  content.className = "content";
  div.appendChild(content);
  if (replay) {
    const btn = document.createElement("button");
    btn.className = "replay";
    btn.setAttribute("aria-label", "Play audio");
    setReplayIcon(btn, "play");
    btn.onclick = () => toggleReplay(div, btn);
    div.appendChild(btn);
  }
  logEl.appendChild(div);
  scrollLog();
  return { div, content };
}

function setReplayIcon(btn, state) {
  if (state === "pause") {
    btn.innerHTML = ICON_PAUSE;
    btn.title = "Pause";
    btn.setAttribute("aria-label", "Pause audio");
    btn.classList.add("playing");
  } else {
    btn.innerHTML = ICON_PLAY;
    btn.title = "Play";
    btn.setAttribute("aria-label", "Play audio");
    btn.classList.remove("playing");
  }
}

// Stop whatever replay is currently playing (unless it's `keep`).
function stopActiveReplay(keep) {
  if (activeReplay && activeReplay.audio !== keep) {
    activeReplay.audio.pause();
    setReplayIcon(activeReplay.btn, "play");
  }
  activeReplay = null;
}

// Play / pause / resume the TTS for a message bubble. The first click fetches
// and synthesizes the audio; later clicks toggle pause/resume on the cached clip.
async function toggleReplay(div, btn) {
  const cached = div._replayAudio;
  if (cached) {
    if (!cached.paused && !cached.ended) {
      cached.pause(); // 'pause' handler restores the play icon
      return;
    }
    if (cached.ended) cached.currentTime = 0;
    stopActiveReplay(cached);
    activeReplay = { audio: cached, btn };
    await cached.play();
    setReplayIcon(btn, "pause");
    return;
  }

  const text = div._ttsText;
  if (!text) return;
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language: div._ttsLang, voice: div._ttsVoice }),
    });
    if (r.ok) {
      const url = URL.createObjectURL(await r.blob());
      const audio = new Audio(url);
      div._replayAudio = audio;
      audio.onended = () => {
        setReplayIcon(btn, "play");
        if (activeReplay && activeReplay.audio === audio) activeReplay = null;
      };
      audio.onpause = () => setReplayIcon(btn, "play");
      stopActiveReplay(audio);
      activeReplay = { audio, btn };
      await audio.play();
      setReplayIcon(btn, "pause");
    }
  } catch (e) {
    /* ignore */
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
}

// Plain-text message (user / status / error). When `tts` ({language, voice}) is
// given, the bubble gets a play/pause button that synthesizes `text`.
function addMessage(text, kind, who, tts) {
  const m = makeMsg(kind, who, !!tts);
  m.content.textContent = text;
  if (tts) {
    m.div._ttsText = text;
    m.div._ttsLang = tts.language;
    m.div._ttsVoice = tts.voice;
  }
  scrollLog(); // makeMsg scrolled while the bubble was still empty; re-scroll now.
  return m;
}

// ---------- Minimal, safe markdown rendering ----------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function renderInline(s) {
  s = escapeHtml(s);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, t, u) => {
    const safe = /^(https?:|mailto:)/i.test(u) ? u : "#";
    return `<a href="${safe}" target="_blank" rel="noopener">${t}</a>`;
  });
  return s;
}

function renderMarkdown(src) {
  const blocks = [];
  src = src.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
    blocks.push(`<pre class="code"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  let html = "";
  let list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of src.split(/\r?\n/)) {
    const ph = raw.match(/^\u0000(\d+)\u0000$/);
    if (ph) { closeList(); html += blocks[+ph[1]]; continue; }
    const head = raw.match(/^(#{1,6})\s+(.*)$/);
    if (head) { closeList(); const lvl = Math.min(head[1].length + 2, 6); html += `<h${lvl}>${renderInline(head[2])}</h${lvl}>`; continue; }
    const ul = raw.match(/^\s*[-*]\s+(.*)$/);
    const ol = raw.match(/^\s*\d+\.\s+(.*)$/);
    if (ul) { if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; } html += `<li>${renderInline(ul[1])}</li>`; continue; }
    if (ol) { if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; } html += `<li>${renderInline(ol[1])}</li>`; continue; }
    closeList();
    if (raw.trim() !== "") html += `<p>${renderInline(raw)}</p>`;
  }
  closeList();
  return html;
}

// ---------- WebSocket ----------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("Ready", "ready");
    uiLocked = false;
    talkBtn.disabled = false;
  };
  ws.onclose = () => {
    setStatus("Disconnected", "error");
    uiLocked = false;
    talkBtn.disabled = true;
    if (vadEnabled) stopVad();
    setTimeout(connect, 1500);
  };
  ws.onerror = () => setStatus("Connection error", "error");

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleControl(JSON.parse(event.data));
    } else {
      playPCM16(event.data, ttsSampleRate);
    }
  };
}

function handleControl(msg) {
  switch (msg.type) {
    case "config":
      ttsSampleRate = msg.tts_sample_rate || 24000;
      setupConfig(msg);
      break;
    case "engine_ready":
      onEngineReady(msg);
      break;
    case "engine_error":
      onEngineError(msg);
      break;
    case "session_saved":
      // A turn was persisted; adopt its id and refresh the sidebar ordering.
      if (msg.session) activeSessionId = msg.session.id;
      refreshSessions();
      break;
    case "session_new":
      activeSessionId = null;
      clearLog();
      markActiveSession();
      break;
    case "session_loaded":
      onSessionLoaded(msg);
      break;
    case "language_update":
      // In "Match my speech" mode the server picked a language from the audio.
      // Keep the dropdown on "Match"; just note the detection in the status.
      if (msg.response) {
        const det = languages.find((l) => l.key === msg.response);
        if (det) setStatus("Detected: " + det.name, "busy");
      }
      break;
    case "user":
      endTurn();
      addMessage(msg.text, "user", "You", { language: msg.language, voice: msg.voice });
      break;
    case "assistant":
      finishTool();
      appendAssistant(msg.text, msg.language, msg.voice);
      break;
    case "tool":
      startTool(msg.name);
      setStatus("Searching...", "busy");
      break;
    case "tool_result":
      showToolResult(msg);
      break;
    case "status":
      addMessage(msg.text, "note");
      break;
    case "error":
      addMessage(msg.text, "error");
      setStatus("Error", "error");
      turnBusy = false;
      break;
    case "done":
      endTurn();
      finishTurn();
      break;
  }
}

// Called when a reply is fully delivered. In hands-free mode we wait for any
// queued TTS to finish playing before listening again, so the bot's own voice
// isn't picked up as a new utterance.
function finishTurn() {
  if (inputMode === "vad" && vadEnabled) {
    const tail = playCtx ? Math.max(0, nextPlayTime - playCtx.currentTime) : 0;
    setStatus("Listening...", "busy");
    setTimeout(() => {
      turnBusy = false;
      vadLastVoice = performance.now();
    }, tail * 1000 + 350);
  } else {
    turnBusy = false;
    setStatus("Ready", "ready");
  }
}

// ---------- Turn / tool rendering ----------
function appendAssistant(text, language, voice) {
  if (!currentBot) {
    currentBot = makeMsg("bot", "Assistant", true);
    currentBotRaw = "";
  }
  currentBotRaw += (currentBotRaw ? " " : "") + text;
  currentBot.content.innerHTML = renderMarkdown(currentBotRaw);
  // Store text + the voice used, so the replay button reproduces this reply.
  currentBot.div._ttsText = currentBotRaw;
  if (language) currentBot.div._ttsLang = language;
  if (voice) currentBot.div._ttsVoice = voice;
  scrollLog();
}

function startTool(name) {
  finishTool();
  const div = document.createElement("div");
  div.className = "msg tool-call";
  div.innerHTML =
    '<div class="tool-head">' +
    '<span class="spinner" aria-hidden="true"></span>' +
    '<svg class="check" viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 4.5l-7 7-3-3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
    '<span class="tool-text">Searching the web</span>' +
    `<span class="tool-name">${escapeHtml(name || "search")}</span>` +
    "</div>";
  logEl.appendChild(div);
  currentTool = div;
  scrollLog();
}

// Linkify URLs in plain tool output; everything else is escaped + preserved.
function renderToolResults(text) {
  return escapeHtml(text).replace(
    /(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>'
  );
}

// Fill the active search blob with the query and a collapsible results preview.
function showToolResult(msg) {
  const div = currentTool;
  if (!div) return;
  const head = div.querySelector(".tool-head");
  head.querySelector(".tool-text").textContent = "Searched the web";

  if (msg.query) {
    let q = head.querySelector(".tool-query");
    if (!q) {
      q = document.createElement("span");
      q.className = "tool-query";
      head.insertBefore(q, head.querySelector(".tool-name"));
    }
    q.textContent = `"${msg.query}"`;
  }

  if (msg.text) {
    let details = div.querySelector(".tool-results");
    if (!details) {
      details = document.createElement("details");
      details.className = "tool-results";
      const summary = document.createElement("summary");
      summary.textContent = "Results preview";
      const body = document.createElement("div");
      body.className = "tool-results-body";
      details.append(summary, body);
      div.appendChild(details);
    }
    details.querySelector(".tool-results-body").innerHTML = renderToolResults(msg.text);
  }
  scrollLog();
}

function finishTool() {
  if (!currentTool) return;
  currentTool.classList.add("done");
  currentTool.querySelector(".tool-text").textContent = "Searched the web";
  currentTool = null;
}

function endTurn() {
  finishTool();
  currentBot = null;
  currentBotRaw = "";
}

// ---------- Session log ----------
// Wipe the transcript view and any in-progress turn state (used when switching
// sessions or starting a new one).
function clearLog() {
  logEl.innerHTML = "";
  endTurn();
  stopActiveReplay(null);
  turnBusy = false;
}

// Render a complete assistant bubble (markdown + replay) from stored history.
function addAssistantBubble(text, language, voice) {
  const m = makeMsg("bot", "Assistant", true);
  m.content.innerHTML = renderMarkdown(text || "");
  m.div._ttsText = text;
  m.div._ttsLang = language;
  m.div._ttsVoice = voice;
  scrollLog();
}

function onSessionLoaded(msg) {
  activeSessionId = msg.id;
  clearLog();
  for (const m of msg.messages || []) {
    if (m.role === "user") {
      addMessage(m.text, "user", "You", { language: m.language, voice: m.voice });
    } else if (m.role === "assistant") {
      addAssistantBubble(m.text, m.language, m.voice);
    }
  }
  markActiveSession();
  setStatus("Ready", "ready");
}

function switchSession(id) {
  if (!wsReady() || id === activeSessionId) return;
  ws.send(JSON.stringify({ type: "load_session", id }));
}

function startNewSession() {
  if (!wsReady()) return;
  ws.send(JSON.stringify({ type: "new_session" }));
}

async function deleteSession(id) {
  try {
    await fetch("/api/sessions/" + encodeURIComponent(id), { method: "DELETE" });
  } catch (e) {
    /* ignore */
  }
  if (id === activeSessionId) {
    // The deleted session was on screen; drop back to a fresh one.
    activeSessionId = null;
    startNewSession();
  }
  refreshSessions();
}

async function refreshSessions() {
  try {
    const data = await (await fetch("/api/sessions")).json();
    renderSessionList(data.sessions || []);
  } catch (e) {
    /* transient; ignore */
  }
}

function renderSessionList(list) {
  sessionListEl.innerHTML = "";
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "session-empty";
    li.textContent = "No saved sessions yet.";
    sessionListEl.appendChild(li);
    return;
  }
  for (const s of list) {
    const li = document.createElement("li");
    li.className = "session-item" + (s.id === activeSessionId ? " active" : "");
    li.dataset.id = s.id;
    li.onclick = () => switchSession(s.id);

    const title = document.createElement("div");
    title.className = "session-title";
    title.textContent = s.title || "Untitled session";

    const meta = document.createElement("div");
    meta.className = "session-meta";
    meta.textContent = formatSessionMeta(s);

    const del = document.createElement("button");
    del.className = "session-del";
    del.type = "button";
    del.title = "Delete session";
    del.innerHTML = "&times;";
    del.onclick = (e) => { e.stopPropagation(); deleteSession(s.id); };

    li.append(title, meta, del);
    sessionListEl.appendChild(li);
  }
}

function formatSessionMeta(s) {
  const turns = s.turns || 0;
  const when = (s.updated_at || "").replace("T", " ").slice(0, 16);
  return `${turns} turn${turns === 1 ? "" : "s"}${when ? " · " + when : ""}`;
}

// Highlight the active session in the sidebar without a full refetch.
function markActiveSession() {
  for (const li of sessionListEl.querySelectorAll(".session-item")) {
    li.classList.toggle("active", li.dataset.id === activeSessionId);
  }
}

// ---------- Language selection ----------
function hasOption(sel, value) {
  return value && Array.from(sel.options).some((o) => o.value === value);
}

function engineName(id) {
  const e = engines.find((x) => x.id === id);
  return e ? e.name : id;
}

// Enable/disable every control from a single place so the "locked while a model
// deploys" state and the "voice is auto in Match mode" rule never fight.
function refreshControls() {
  const matchMode = respLangEl.value === "match";
  ttsEngineEl.disabled = uiLocked;
  sttLangEl.disabled = uiLocked;
  respLangEl.disabled = uiLocked;
  voiceEl.disabled = uiLocked || matchMode;
  talkBtn.disabled = uiLocked || !wsReady();
}

function setupConfig(msg) {
  const firstSetup = engines.length === 0;
  engines = msg.engines || [];
  if (firstSetup) {
    ttsEngineEl.innerHTML = "";
    for (const e of engines) ttsEngineEl.add(new Option(e.name, e.id));
    ttsEngineEl.onchange = () => switchEngine(ttsEngineEl.value);
    sttLangEl.onchange = sendLanguage;
    respLangEl.onchange = () => { populateVoices(); sendLanguage(); };
    voiceEl.onchange = sendLanguage;
  }
  activeEngine = msg.engine;
  ttsEngineEl.value = activeEngine;

  // STT languages are engine-agnostic (whisper); preserve any prior choice.
  const prevStt = sttLangEl.value;
  sttLanguages = msg.stt_languages || [];
  sttLangEl.innerHTML = '<option value="auto">Auto-detect</option>';
  for (const l of sttLanguages) sttLangEl.add(new Option(l.name, l.key));
  sttLangEl.value = hasOption(sttLangEl, prevStt) && !firstSetup ? prevStt : (msg.default_stt || "auto");

  applyTtsLanguages(msg.tts_languages, msg.default_response || "match");
  uiLocked = false;
  refreshControls();
  sendLanguage();
}

// Rebuild the reply-language + voice dropdowns from the active engine's language
// set, keeping the current selection when the new engine still supports it.
function applyTtsLanguages(list, fallbackResp) {
  const prevResp = respLangEl.value;
  const prevVoice = voiceEl.value;
  languages = list || [];
  respLangEl.innerHTML = '<option value="match">Match my speech</option>';
  for (const lang of languages) respLangEl.add(new Option(lang.name, lang.key));
  respLangEl.value = hasOption(respLangEl, prevResp) ? prevResp : (fallbackResp || "match");
  populateVoices();
  if (hasOption(voiceEl, prevVoice)) voiceEl.value = prevVoice;
}

function populateVoices() {
  const lang = languages.find((l) => l.key === respLangEl.value);
  voiceEl.innerHTML = "";
  if (!lang) {
    // "Match my speech": voice is auto-selected per detected language.
    voiceEl.add(new Option("(auto)", ""));
  } else {
    for (const v of lang.voices) voiceEl.add(new Option(v, v));
    voiceEl.value = lang.voices[0];
  }
  refreshControls();
}

// ---- Live TTS-engine switching (locks the UI until the model is deployed) ----
function switchEngine(id) {
  if (!wsReady() || id === activeEngine) return;
  uiLocked = true;
  refreshControls();
  setStatus("Loading " + engineName(id) + "...", "busy");
  ws.send(JSON.stringify({ type: "set_engine", engine: id }));
}

function onEngineReady(msg) {
  ttsSampleRate = msg.tts_sample_rate || ttsSampleRate;
  activeEngine = msg.engine;
  ttsEngineEl.value = activeEngine;
  applyTtsLanguages(msg.tts_languages, respLangEl.value || "match");
  uiLocked = false;
  refreshControls();
  setStatus("Ready", "ready");
  addMessage(engineName(activeEngine) + " voice engine ready.", "note");
  sendLanguage();
}

function onEngineError(msg) {
  addMessage("Couldn't load " + engineName(msg.engine) + ": " + (msg.text || "unknown error"), "error");
  ttsEngineEl.value = activeEngine; // revert to the engine still deployed
  uiLocked = false;
  refreshControls();
  setStatus("Ready", "ready");
}

function sendLanguage() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(
    JSON.stringify({
      type: "set_language",
      stt: sttLangEl.value,
      response: respLangEl.value,
      voice: voiceEl.value,
    })
  );
}

// ---------- Live metrics (GPU + context tokens) ----------
function setBar(barEl, pct) {
  barEl.style.width = Math.max(0, Math.min(100, pct)) + "%";
  barEl.classList.toggle("warn", pct >= 75 && pct < 90);
  barEl.classList.toggle("crit", pct >= 90);
}

async function pollMetrics() {
  try {
    const m = await (await fetch("/api/metrics")).json();
    if (m.gpu) {
      const g = m.gpu;
      const pct = Math.round((100 * g.mem_used_mb) / g.mem_total_mb);
      setBar(gpuBarEl, pct);
      gpuTextEl.textContent =
        `${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(1)} GB · ${g.util_pct}%`;
    } else {
      gpuTextEl.textContent = "n/a";
    }
    if (m.tokens) {
      const t = m.tokens;
      const max = t.max || 0;
      setBar(ctxBarEl, max ? (100 * t.current) / max : 0);
      ctxTextEl.textContent = `${t.current} / ${max || "?"} tokens`;
    }
  } catch (e) {
    /* transient; ignore */
  }
}

setInterval(pollMetrics, 2000);
pollMetrics();

// ---------- Microphone capture ----------
function wsReady() {
  return ws && ws.readyState === WebSocket.OPEN;
}

async function ensureCapture() {
  if (audioCtx) {
    if (audioCtx.state === "suspended") await audioCtx.resume();
    return;
  }
  audioCtx = new AudioContext();
  captureRate = audioCtx.sampleRate;
  await audioCtx.audioWorklet.addModule("/static/worklet.js");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, "capture-processor");
  workletNode.port.onmessage = (e) => {
    const frame = e.data;
    if (recording) captureBuffers.push(frame);
    if (inputMode === "vad" && vadEnabled) processVadFrame(frame);
  };
  source.connect(workletNode);
  // Worklet has no audible output; do not connect to destination.
}

// Returns true once the mic is live, false (with a message) if access failed.
async function ensureMicReady() {
  try {
    await ensureCapture();
    return true;
  } catch (err) {
    addMessage("Microphone access failed: " + err.message, "error");
    return false;
  }
}

// Begin accumulating audio. `seed` optionally pre-fills the buffer (VAD pre-roll).
function beginCapture(seed) {
  captureBuffers = seed && seed.length ? seed.slice() : [];
  recording = true;
  talkBtn.classList.add("recording");
  setStatus("Listening...", "busy");
}

// Finalize the current capture and ship it as one utterance, or bail if empty.
function sendCapture() {
  recording = false;
  talkBtn.classList.remove("recording");
  const float = mergeBuffers(captureBuffers);
  captureBuffers = [];
  if (float.length === 0 || !wsReady()) {
    finishTurn();
    return;
  }
  const pcm = floatToPCM16(downsample(float, captureRate, TARGET_RATE));
  ws.send(pcm);
  turnBusy = true;
  setStatus("Thinking...", "busy");
}

// ---------- Push-to-talk ----------
async function pushStart() {
  if (inputMode !== "push" || recording || !wsReady()) return;
  if (!(await ensureMicReady())) return;
  beginCapture(null);
}

function pushStop() {
  if (inputMode !== "push" || !recording) return;
  sendCapture();
}

// ---------- Threshold (hands-free) ----------
async function startVad() {
  if (vadEnabled || !wsReady()) return;
  if (!(await ensureMicReady())) return;
  vadEnabled = true;
  vadActive = false;
  turnBusy = false;
  vadLevelDb = -100;
  prerollFrames = [];
  updateModeUI();
  setStatus("Listening...", "busy");
  requestAnimationFrame(meterLoop);
}

function stopVad() {
  vadEnabled = false;
  if (recording) {
    recording = false;
    talkBtn.classList.remove("recording");
    captureBuffers = [];
  }
  vadActive = false;
  prerollFrames = [];
  meterFillEl.style.width = "0%";
  meterFillEl.classList.remove("hot");
  updateModeUI();
  setStatus(wsReady() ? "Ready" : "Disconnected", wsReady() ? "ready" : "error");
}

function frameDb(frame) {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  const rms = Math.sqrt(sum / frame.length);
  return rms > 1e-7 ? 20 * Math.log10(rms) : -100;
}

// Per-frame VAD state machine: opens a segment when the level crosses the
// threshold and closes it after HANGOVER_MS of silence.
function processVadFrame(frame) {
  const db = frameDb(frame);
  vadLevelDb = vadLevelDb * VAD.SMOOTH + db * (1 - VAD.SMOOTH);

  // Keep a short rolling pre-roll so the start of a word isn't clipped.
  prerollFrames.push(frame);
  let preSamples = prerollFrames.reduce((n, f) => n + f.length, 0);
  const maxPre = (VAD.PREROLL_MS / 1000) * captureRate;
  while (preSamples - prerollFrames[0].length >= maxPre) {
    preSamples -= prerollFrames.shift().length;
  }

  const now = performance.now();
  const isVoice = db > vadThreshold;
  if (isVoice) {
    vadLastVoice = now;
    if (!vadActive && !turnBusy) {
      vadActive = true;
      vadSegStart = now;
      beginCapture(prerollFrames);
    }
  } else if (vadActive && now - vadLastVoice > VAD.HANGOVER_MS) {
    vadActive = false;
    if (now - vadSegStart >= VAD.MIN_SPEECH_MS) {
      sendCapture();
    } else {
      recording = false;
      talkBtn.classList.remove("recording");
      captureBuffers = [];
      setStatus("Listening...", "busy");
    }
  }
}

function meterLoop() {
  if (!vadEnabled) return;
  const pct = Math.max(0, Math.min(100, ((vadLevelDb + 80) / 80) * 100));
  meterFillEl.style.width = pct + "%";
  meterFillEl.classList.toggle("hot", vadLevelDb > vadThreshold);
  requestAnimationFrame(meterLoop);
}

function mergeBuffers(buffers) {
  let total = 0;
  for (const b of buffers) total += b.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const b of buffers) {
    out.set(b, offset);
    offset += b.length;
  }
  return out;
}

function downsample(buffer, inRate, outRate) {
  if (outRate >= inRate) return buffer;
  const ratio = inRate / outRate;
  const newLen = Math.floor(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, buffer.length - 1);
    const frac = idx - i0;
    result[i] = buffer[i0] * (1 - frac) + buffer[i1] * frac;
  }
  return result;
}

function floatToPCM16(f32) {
  const view = new DataView(new ArrayBuffer(f32.length * 2));
  for (let i = 0; i < f32.length; i++) {
    let s = Math.max(-1, Math.min(1, f32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view.buffer;
}

// ---------- Playback ----------
function playPCM16(arrayBuffer, rate) {
  if (!playCtx) playCtx = new AudioContext();
  if (playCtx.state === "suspended") playCtx.resume();
  const view = new DataView(arrayBuffer);
  const n = Math.floor(arrayBuffer.byteLength / 2);
  if (n === 0) return;
  const f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = view.getInt16(i * 2, true) / 32768;

  const buf = playCtx.createBuffer(1, n, rate);
  buf.copyToChannel(f32, 0);
  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);

  const now = playCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  src.start(nextPlayTime);
  nextPlayTime += buf.duration;
}

// ---------- Input wiring ----------
function updateModeUI() {
  vadPanelEl.hidden = inputMode !== "vad";
  if (inputMode === "vad") {
    talkBtn.classList.toggle("listening", vadEnabled);
    talkBtn.classList.remove("recording");
    talkLabelEl.textContent = vadEnabled ? "Stop listening" : "Start listening";
    hintEl.textContent = vadEnabled
      ? "Listening hands-free - just speak. Tune the threshold so the meter only fills past the marker while you talk."
      : "Press Start, then speak whenever you like. Recording begins automatically once your voice passes the threshold.";
  } else {
    talkBtn.classList.remove("listening", "recording");
    talkLabelEl.textContent = "Hold to talk";
    hintEl.textContent = "Hold the button (or press and hold the spacebar) while speaking, then release.";
  }
}

function setMode(mode) {
  if (mode === inputMode) return;
  if (inputMode === "vad" && vadEnabled) stopVad();
  if (recording) {
    recording = false;
    talkBtn.classList.remove("recording");
    captureBuffers = [];
  }
  inputMode = mode;
  updateModeUI();
  if (wsReady()) setStatus("Ready", "ready");
}

function updateThreshold() {
  vadThreshold = Number(threshEl.value);
  threshTextEl.textContent = vadThreshold + " dB";
  meterThreshEl.style.left = ((vadThreshold + 80) / 80) * 100 + "%";
}

modeRadios.forEach((r) => r.addEventListener("change", () => { if (r.checked) setMode(r.value); }));
threshEl.addEventListener("input", updateThreshold);

// Button: hold-to-talk in push mode, start/stop toggle in hands-free mode.
talkBtn.addEventListener("mousedown", pushStart);
talkBtn.addEventListener("mouseup", pushStop);
talkBtn.addEventListener("mouseleave", pushStop);
talkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); pushStart(); });
talkBtn.addEventListener("touchend", (e) => {
  e.preventDefault();
  if (inputMode === "vad") { toggleVad(); return; }
  pushStop();
});
talkBtn.addEventListener("click", () => { if (inputMode === "vad") toggleVad(); });

function toggleVad() {
  if (vadEnabled) stopVad(); else startVad();
}

// Spacebar mirrors the hold-to-talk button (push mode only).
document.addEventListener("keydown", (e) => {
  if (inputMode === "push" && e.code === "Space" && !e.repeat && !talkBtn.disabled) {
    e.preventDefault();
    pushStart();
  }
});
document.addEventListener("keyup", (e) => {
  if (inputMode === "push" && e.code === "Space") {
    e.preventDefault();
    pushStop();
  }
});

newSessionBtn.addEventListener("click", startNewSession);
sidebarToggle.addEventListener("click", () => sidebarEl.classList.toggle("hidden"));
// The sidebar is an overlay on narrow screens; start it collapsed there.
if (window.matchMedia("(max-width: 640px)").matches) sidebarEl.classList.add("hidden");

updateThreshold();
updateModeUI();
connect();
refreshSessions();

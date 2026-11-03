"use strict";

const TEXT = {
  es: {
    nav_try: "Pruébalo", nav_results: "Resultados", nav_code: "Código",
    tagline: "Una memoria para asistentes de IA que olvida de verdad cuando se lo pides.",
    intro: "Escribe como hablarías con un asistente. INFINITO y una memoria normal reciben lo mismo y responden con el mismo modelo. Ninguno ve el chat: solo lo que recuerda.",
    try_label: "Prueba esta secuencia",
    try_hint: "Cuéntale dos cosas, intenta darle una contraseña y pídele que olvide una. Después mira qué sigue guardando cada memoria.",
    placeholder: "Escribe un mensaje…", send: "Enviar",
    left: (n) => (n === 1 ? "Te queda 1 mensaje" : `Te quedan ${n} mensajes`),
    inf_sub: "Guarda lo que dices, bloquea secretos y borra cuando se lo pides.",
    plain_name: "Memoria normal",
    plain_sub: "Guarda cada mensaje y recupera los más parecidos, como hacen la mayoría.",
    remembers: "Lo que recuerda", nothing: "Nada todavía.",
    secret_note: "secreto guardado tal cual",
    reset: "Borrar todo y empezar de cero",
    thinking: "Pensando…",
    no_answer: "El modelo no respondió. Inténtalo de nuevo.",
    erased: (n) => (n === 0 ? "Petición de olvido: no había nada que borrar." : n === 1
      ? "Petición de olvido: 1 recuerdo borrado de la base de datos."
      : `Petición de olvido: ${n} recuerdos borrados de la base de datos.`),
    kept_request: "La petición se guarda como un recuerdo más; no borra nada.",
    blocked: (what) => `Bloqueado: parece ${what}. No se guarda ni se envía al modelo.`,
    secret_kept: "Guardado tal cual. Esta demo no se lo envía al modelo, pero una memoria normal sí lo haría.",
    secret_kinds: { password: "una contraseña", pin: "un PIN", api_key: "una clave de API", private_key: "una clave privada", secret: "un secreto" },
    errors: {
      budget: "La demo ha llegado a su límite de gasto de hoy. Vuelve mañana, o mira el código en GitHub.",
      visitor_limit: "Has llegado al límite de mensajes de esta demo. Pulsa «Borrar todo y empezar de cero» para seguir.",
      ip_limit: "Demasiados mensajes desde tu conexión en la última hora. Espera un poco.",
      busy: "Hay mucha gente probando la demo ahora mismo. Inténtalo en unos segundos.",
      unavailable: "La demo se está preparando. Mientras tanto, mira los resultados.",
      too_long: "El mensaje es demasiado largo.",
      empty: "Escribe algo primero.",
      network: "No hay conexión con la demo. Inténtalo de nuevo.",
    },
    expired: "Tu sesión anterior caducó: empezamos de cero.",
    nudge: "Hasta aquí las dos memorias se comportan igual. Ahora pídele que olvide algo de lo que le has contado («olvida lo de…») y compara qué sigue guardando cada una.",
    offline: "Modo sin conexión: las respuestas son simuladas, pero el borrado es real.",
    privacy: (model, minutes) => `Tus mensajes solo existen en la memoria de esta demo, sin escribirse en disco, y se borran al recargar o tras ${minutes} minutos sin uso. No se guarda ningún registro de lo que escribes.${model ? ` Las respuestas las genera ${model} de OpenAI.` : ""}`,
    examples: [
      "Vivo en Valencia.",
      "Trabajo de enfermera en un hospital.",
      "Mi contraseña del banco es hunter2",
      "Olvida dónde vivo.",
      "Haz una lista de todo lo que sabes de mí.",
      "¿En qué trabajo?",
    ],
    results_title: "Resultados",
    results_intro: "Bemula es un benchmark público de olvido a petición: a mitad de una conversación el usuario pide olvidar algo y después se comprueba si de verdad se ha olvidado, también con preguntas trampa. INFINITO se ejecutó una sola vez en Bemula-Lite, con el mismo modelo y el mismo juez que los sistemas publicados en el paper.",
    col_system: "Sistema", col_score: "Puntuación", col_passed: "Peticiones superadas", col_efficacy: "Olvido",
    col_attack: "Ante ataques", col_retention: "Retención", col_preservation: "Preservación",
    results_note: "Lejos de estar resuelto: en el 44 % de las preguntas sobre algo olvidado, la respuesta todavía lo dejó ver. En esta demo, con pocos mensajes, el modelo suele respetar la petición aunque la memoria normal conserve el dato; en conversaciones largas como las de Bemula se filtra mucho más: un RAG simple solo resistió el 14 % de las preguntas trampa.",
    updates: "Con datos que cambian («ahora vivo en Valencia») y un LLM real, INFINITO acierta 112 de 120 escenarios frente a 117 de un RAG que lo guarda todo, enviando un 31 % menos de contexto.",
    link_bemula: "Ejecución de Bemula-Lite", link_updates: "Confirmación con LLM real", link_method: "Metodología completa",
    how_title: "Cómo olvida",
    how1_t: "Reconoce la petición.", how1: "Unas reglas detectan «olvida», «borra», «deja de guardar»… antes de que el mensaje se guarde.",
    how2_t: "Elige qué borrar.", how2: "Un LLM revisa todos los recuerdos guardados y señala los que contienen ese dato.",
    how3_t: "Borra de verdad.", how3: "Las filas se eliminan con su historial y se sobrescriben en la base de datos. Solo queda un contador.",
    footer: "INFINITO es open source (MIT).",
    names: { "Full context": "Contexto completo", "Naive RAG": "RAG simple" },
  },
  en: {
    nav_try: "Try it", nav_results: "Results", nav_code: "Code",
    tagline: "Memory for AI assistants that truly forgets when you ask it to.",
    intro: "Write as you would to an assistant. INFINITO and a plain memory get the same messages and answer with the same model. Neither sees the chat: only what it remembers.",
    try_label: "Try this sequence",
    try_hint: "Tell it two things, try to give it a password and ask it to forget one. Then look at what each memory still keeps.",
    placeholder: "Type a message…", send: "Send",
    left: (n) => (n === 1 ? "1 message left" : `${n} messages left`),
    inf_sub: "Keeps what you say, blocks secrets and deletes when you ask.",
    plain_name: "Plain memory",
    plain_sub: "Keeps every message and retrieves the most similar ones, like most do.",
    remembers: "What it remembers", nothing: "Nothing yet.",
    secret_note: "secret kept as is",
    reset: "Erase everything and start over",
    thinking: "Thinking…",
    no_answer: "The model did not answer. Try again.",
    erased: (n) => (n === 0 ? "Forget request: there was nothing to delete." : n === 1
      ? "Forget request: 1 memory deleted from the database."
      : `Forget request: ${n} memories deleted from the database.`),
    kept_request: "The request is stored as one more memory; nothing is deleted.",
    blocked: (what) => `Blocked: this looks like ${what}. It is not stored or sent to the model.`,
    secret_kept: "Stored as is. This demo does not send it to the model, but a plain memory would.",
    secret_kinds: { password: "a password", pin: "a PIN", api_key: "an API key", private_key: "a private key", secret: "a secret" },
    errors: {
      budget: "The demo has reached today's spending limit. Come back tomorrow, or read the code on GitHub.",
      visitor_limit: "You have reached this demo's message limit. Press “Erase everything and start over” to go on.",
      ip_limit: "Too many messages from your connection in the last hour. Please wait a little.",
      busy: "Many people are trying the demo right now. Try again in a few seconds.",
      unavailable: "The demo is getting ready. Meanwhile, have a look at the results.",
      too_long: "The message is too long.",
      empty: "Write something first.",
      network: "The demo cannot be reached. Try again.",
    },
    expired: "Your earlier session expired: starting over.",
    nudge: "So far both memories behave the same. Now ask it to forget something you told it (“forget what I said about…”) and compare what each one still keeps.",
    offline: "Offline mode: answers are simulated, but deletion is real.",
    privacy: (model, minutes) => `Your messages exist only in this demo's memory, never written to disk, and are erased when you reload or after ${minutes} idle minutes. No record of what you write is kept.${model ? ` Answers come from OpenAI's ${model}.` : ""}`,
    examples: [
      "I live in Valencia.",
      "I work as a nurse at a hospital.",
      "My bank password is hunter2",
      "Forget where I live.",
      "List everything you know about me.",
      "What do I do for work?",
    ],
    results_title: "Results",
    results_intro: "Bemula is a public benchmark of forgetting on request: mid-conversation the user asks the assistant to forget something, and the system is then checked to see whether it really forgot, trick questions included. INFINITO ran once on Bemula-Lite, with the same model and the same judge as the systems published in the paper.",
    col_system: "System", col_score: "Score", col_passed: "Requests passed", col_efficacy: "Forgetting",
    col_attack: "Under attack", col_retention: "Retention", col_preservation: "Preservation",
    results_note: "Far from solved: in 44% of the questions about something forgotten, the answer still gave it away. In this demo, with few messages, the model usually honors the request even though the plain memory keeps the fact; in long conversations like Bemula's it leaks far more: a simple RAG resisted only 14% of the trick questions.",
    updates: "With facts that change (“I live in Valencia now”) and a real LLM, INFINITO gets 112 of 120 scenarios right against 117 for a RAG that stores everything, while sending 31% less context.",
    link_bemula: "Bemula-Lite run", link_updates: "Real-LLM confirmation", link_method: "Full methodology",
    how_title: "How it forgets",
    how1_t: "It recognizes the request.", how1: "Rules spot “forget”, “delete”, “stop keeping”… before the message is stored.",
    how2_t: "It picks what to delete.", how2: "An LLM reviews every stored memory and points out the ones that hold that information.",
    how3_t: "It deletes for real.", how3: "The rows are removed with their history and overwritten in the database. Only a count remains.",
    footer: "INFINITO is open source (MIT).",
    names: {},
  },
};

// Bemula-Lite, from the published run (README, Evaluation).
const BEMULA = [
  ["INFINITO 0.5", 45.9, 3, 55.8, 50.1, 98.1, 81.7],
  ["A-MEM", 26.3, 0, 35.6, 28.9, 86.5, 74.2],
  ["Full context", 23.1, 0, 26.0, 28.5, 89.4, 75.3],
  ["LangMem", 20.1, 0, 48.1, 35.4, 66.3, 30.1],
  ["LightMem", 18.1, 0, 22.1, 19.0, 98.1, 90.3],
  ["SimpleMem", 17.5, 0, 17.3, 22.2, 96.2, 84.9],
  ["Naive RAG", 13.1, 0, 17.3, 14.1, 84.6, 79.6],
  ["Mem0", 5.7, 0, 10.6, 4.2, 96.2, 84.9],
];

const $ = (id) => document.getElementById(id);
const state = {
  lang: "es",
  session: null,
  left: null,
  busy: false,
  status: null,
  seen: { inf: new Set(), plain: new Set() },
  // Turns since the visitor last saw INFINITO differ: a forget request or a blocked secret.
  sameTurns: 0,
  nudged: false,
};
const NUDGE_AFTER = 3;

function pickLanguage() {
  let saved = null;
  try { saved = localStorage.getItem("infinito-demo-lang"); } catch (e) { /* storage blocked */ }
  if (saved === "es" || saved === "en") return saved;
  return (navigator.language || "es").toLowerCase().startsWith("es") ? "es" : "en";
}

function t(key) { return TEXT[state.lang][key]; }

function applyLanguage() {
  const lang = state.lang;
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const value = TEXT[lang][el.dataset.i18n];
    if (typeof value === "string") el.textContent = value;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = TEXT[lang][el.dataset.i18nPlaceholder];
  });
  $("lang").textContent = lang === "es" ? "EN" : "ES";
  renderExamples();
  renderTable();
  renderLeft();
  renderStatus();
}

function renderExamples() {
  const box = $("examples");
  box.replaceChildren();
  t("examples").forEach((text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => { button.classList.add("used"); send(text); });
    box.append(button);
  });
}

function number(value, digits = 1) {
  return value.toLocaleString(state.lang === "es" ? "es-ES" : "en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function renderTable() {
  const body = document.querySelector("#bemula tbody");
  body.replaceChildren();
  const names = t("names");
  BEMULA.forEach(([name, score, passed, eff, attack, ret, pres], index) => {
    const row = document.createElement("tr");
    if (index === 0) row.className = "ours";
    const cells = [names[name] || name, number(score), `${passed} / 62`, number(eff), number(attack), number(ret), number(pres)];
    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  });
}

function renderLeft() {
  $("left").textContent = state.left === null ? "" : t("left")(state.left);
}

function renderStatus() {
  const status = state.status;
  const notice = $("notice");
  notice.hidden = true;
  if (!status) return;
  if (!status.available) {
    notice.textContent = t("errors").unavailable;
    notice.hidden = false;
  } else if (status.mode === "offline") {
    notice.textContent = t("offline");
    notice.hidden = false;
  }
  $("privacy").textContent = t("privacy")(status.model, status.session_minutes);
  $("send").disabled = !status.available || state.busy;
}

async function api(path, body) {
  let response;
  try {
    response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw Object.assign(new Error("network"), { code: "network" });
  }
  let data = {};
  try { data = await response.json(); } catch (e) { /* not JSON */ }
  if (!response.ok) throw Object.assign(new Error(data.error || "network"), { code: data.error || "network" });
  return data;
}

async function openSession() {
  const data = await api("/api/session", {});
  state.session = data.session;
  state.left = data.messages_left;
  renderLeft();
}

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function addTurn(text) {
  $("empty").hidden = true;
  const turn = element("article", "turn");
  turn.append(element("p", "you", text));
  const answers = element("div", "answers");
  const inf = element("div", "answer inf pending");
  inf.append(element("span", "who", "INFINITO"), element("p", "", t("thinking")));
  const plain = element("div", "answer plain pending");
  plain.append(element("span", "who", t("plain_name")), element("p", "", t("thinking")));
  answers.append(inf, plain);
  turn.append(answers);
  $("timeline").append(turn);
  turn.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  return { turn, inf, plain };
}

function fillAnswer(box, text, event, eventClass) {
  box.classList.remove("pending");
  if (text === null) box.querySelector("p").remove();
  else box.querySelector("p").textContent = text;
  if (event) box.append(element("p", `event ${eventClass}`, event));
}

function fillTurn(parts, data, text) {
  const inf = data.infinito;
  const plain = data.plain;
  if (inf.blocked) {
    const kind = t("secret_kinds")[inf.blocked] || t("secret_kinds").secret;
    fillAnswer(parts.inf, null, t("blocked")(kind), "blocked");
    fillAnswer(parts.plain, null, t("secret_kept"), "kept");
    return;
  }
  const forgot = typeof inf.forgotten === "number";
  fillAnswer(parts.inf, inf.answer || t("no_answer"), forgot ? t("erased")(inf.forgotten) : null, "erased");
  fillAnswer(parts.plain, plain.answer || t("no_answer"), forgot && plain.stored ? t("kept_request") : null, "kept");
}

function failTurn(parts, code) {
  parts.turn.classList.add("error");
  parts.turn.append(element("p", "turn-error", t("errors")[code] || t("errors").network));
}

function clock(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString(state.lang === "es" ? "es-ES" : "en-GB", { hour: "2-digit", minute: "2-digit" });
}

function renderRows(kind, rows) {
  const list = $(`${kind}-rows`);
  const seen = state.seen[kind];
  const now = new Set(rows.map((r) => r.id));
  // Rows that disappeared are struck through, then fade out.
  list.querySelectorAll("li[data-id]").forEach((li) => {
    if (!now.has(li.dataset.id) && !li.classList.contains("erasing")) {
      li.classList.add("erasing");
      setTimeout(() => li.classList.add("gone"), 1400);
      setTimeout(() => li.remove(), 2000);
    }
  });
  list.querySelectorAll("li.none").forEach((li) => li.remove());
  rows.forEach((row) => {
    if (list.querySelector(`li[data-id="${CSS.escape(row.id)}"]`)) return;
    const li = element("li", row.secret ? "secret" : "");
    li.dataset.id = row.id;
    const time = element("time", "", clock(row.at));
    time.dateTime = row.at;
    const text = element("span", "", row.text);
    if (row.secret) text.dataset.note = t("secret_note");
    li.append(time, text);
    if (!seen.has(row.id)) {
      li.classList.add("fresh");
      setTimeout(() => li.classList.remove("fresh"), 1600);
    }
    seen.add(row.id);
    list.append(li);
  });
  if (!rows.length && !list.querySelector("li[data-id]:not(.erasing)")) {
    const none = element("li", "none");
    none.append(element("span", "none", t("nothing")));
    list.append(none);
  }
  $(`${kind}-count`).textContent = String(rows.length);
}

async function send(text) {
  text = (text || "").trim();
  if (!text || state.busy || (state.status && !state.status.available)) return;
  state.busy = true;
  $("send").disabled = true;
  const parts = addTurn(text);
  try {
    if (!state.session) await openSession();
    let data;
    try {
      data = await api("/api/chat", { session: state.session, message: text });
    } catch (error) {
      if (error.code !== "unknown_session") throw error;
      clearMemories();
      parts.turn.before(element("p", "muted", t("expired")));
      await openSession();
      data = await api("/api/chat", { session: state.session, message: text });
    }
    fillTurn(parts, data, text);
    renderRows("inf", data.infinito.memories);
    renderRows("plain", data.plain.memories);
    nudgeTowardsForgetting(data);
    state.left = data.messages_left;
    renderLeft();
  } catch (error) {
    failTurn(parts, error.code);
  } finally {
    state.busy = false;
    $("send").disabled = !!(state.status && !state.status.available);
  }
}

function nudgeTowardsForgetting(data) {
  const differed = data.infinito.blocked || typeof data.infinito.forgotten === "number";
  state.sameTurns = differed ? 0 : state.sameTurns + 1;
  if (differed) state.nudged = true;
  if (state.nudged || state.sameTurns < NUDGE_AFTER) return;
  state.nudged = true;
  const note = element("p", "nudge", t("nudge"));
  $("timeline").append(note);
  note.scrollIntoView({ block: "nearest" });
}

function clearMemories() {
  ["inf", "plain"].forEach((kind) => {
    $(`${kind}-rows`).replaceChildren();
    state.seen[kind].clear();
    renderRows(kind, []);
  });
}

async function resetAll() {
  const old = state.session;
  state.session = null;
  state.left = null;
  if (old) {
    try { await api("/api/forget-everything", { session: old }); } catch (e) { /* the server drops it after an hour anyway */ }
  }
  $("timeline").querySelectorAll(".turn, p.muted, p.nudge").forEach((el) => el.remove());
  state.sameTurns = 0;
  state.nudged = false;
  $("empty").hidden = false;
  document.querySelectorAll(".examples button.used").forEach((b) => b.classList.remove("used"));
  clearMemories();
  renderLeft();
}

async function boot() {
  state.lang = pickLanguage();
  applyLanguage();
  clearMemories();
  $("lang").addEventListener("click", () => {
    state.lang = state.lang === "es" ? "en" : "es";
    try { localStorage.setItem("infinito-demo-lang", state.lang); } catch (e) { /* storage blocked */ }
    applyLanguage();
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const box = $("message");
    const text = box.value;
    box.value = "";
    send(text);
  });
  $("message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("reset").addEventListener("click", resetAll);
  try {
    state.status = await api("/api/status");
    $("message").maxLength = state.status.max_chars;
  } catch (e) {
    state.status = null;
  }
  renderStatus();
}

boot();

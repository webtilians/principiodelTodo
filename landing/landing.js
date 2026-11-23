/* INFINITO interactive landing.
   Draws the scenes of escenas.js around the brain, the wires that join them and, when a scene is pressed, plays its
   flow: the data travels to the brain, is remembered, analysed and, if the scene has an output, goes back as a proposal
   or an action. A classic script, not a module, so the page also opens from a file on a laptop with no connection.
   ?expo in the address plays the tour in a loop, for a screen at a stand; a touch pauses it for 45 seconds. */
(function () {
  "use strict";

  const CFG = window.INFINITO_LANDING;
  const stage = document.getElementById("escenario");
  if (!CFG || !stage) return;

  const SVGNS = "http://www.w3.org/2000/svg";
  const XLINK = "http://www.w3.org/1999/xlink";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const narrow = window.matchMedia("(max-width: 760px)");
  const expo = new URLSearchParams(window.location.search).has("expo");
  if (expo) document.documentElement.classList.add("expo");
  /* Reduced motion stills the decoration (stars, sparks); a pressed scene still plays, its journey is the content. */
  const calm = () => reduceMotion.matches && !expo;

  const wires = stage.querySelector(".wires");
  const core = document.getElementById("core");
  const ripples = core.querySelector(".ripples");
  const countEl = document.getElementById("count");
  const appsEl = stage.querySelector(".apps");
  const memlist = document.getElementById("memlist");
  const tourBtn = document.getElementById("tour");
  const panel = {
    badge: document.getElementById("panel-badge"),
    title: document.getElementById("panel-title"),
    text: document.getElementById("panel-text"),
    steps: document.getElementById("steps"),
    analysis: document.getElementById("analysis"),
    result: document.getElementById("panel-result"),
  };

  const ART = {
    persona:
      '<svg class="art" viewBox="0 0 120 90" aria-hidden="true">' +
      '<rect class="ln soft" x="8" y="22" width="78" height="56" rx="10"/>' +
      '<circle class="ln" cx="32" cy="43" r="9"/>' +
      '<path class="ln" d="M18 69c2.5-8.5 8-12.5 14-12.5S43.5 60.5 46 69"/>' +
      '<path class="ln dim" d="M56 40h20M56 50h14M56 60h18"/>' +
      '<g class="speech"><path class="ln acc soft" d="M78 4h30a6 6 0 0 1 6 6v12a6 6 0 0 1-6 6H94l-7 7v-7h-9a6 6 0 0 1-6-6V10a6 6 0 0 1 6-6z"/>' +
      '<path class="ln acc" d="M82 13h22M82 19h14"/></g></svg>',
    gafas:
      '<svg class="art" viewBox="0 0 120 90" aria-hidden="true">' +
      '<path class="ln dim" d="M6 36l12-3M114 36l-12-3"/>' +
      '<rect class="ln soft" x="17" y="28" width="36" height="27" rx="12"/>' +
      '<rect class="ln soft" x="67" y="28" width="36" height="27" rx="12"/>' +
      '<path class="ln" d="M53 37c4-5 10-5 14 0"/>' +
      '<path class="ln dim" d="M25 39l9-4M75 39l9-4"/>' +
      '<circle class="accf" cx="23" cy="24" r="3.2"/>' +
      '<path class="ln acc" d="M40 72h40a20 9 0 0 1-40 0z"/>' +
      '<path class="ln" d="M46 70c4-5 7 1 11-2.5s7 1 11-2 6 1 8-1"/>' +
      '<circle class="flash" cx="23" cy="24" r="18"/></svg>',
    portatil:
      '<svg class="art" viewBox="0 0 120 90" aria-hidden="true">' +
      '<rect class="ln soft" x="22" y="8" width="76" height="54" rx="6"/>' +
      '<path class="ln" d="M10 70h100l-7 9H17z"/>' +
      '<rect class="ln dim" x="31" y="18" width="7" height="7" rx="1.5"/><path class="ln dim" d="M43 21.5h24"/>' +
      '<path class="ln" d="M32.5 21.5l2 2 3-3.5"/>' +
      '<rect class="ln dim" x="31" y="30" width="7" height="7" rx="1.5"/><path class="ln dim" d="M43 33.5h18"/>' +
      '<rect class="accf decision" x="31" y="44" width="38" height="9" rx="4.5"/>' +
      '<g class="toast"><rect class="toastbg" x="58" y="13" width="35" height="21" rx="5"/>' +
      '<path class="ln acc" d="M63 20.5h24M63 27h15"/></g></svg>',
    jardin:
      '<svg class="art" viewBox="0 0 120 90" aria-hidden="true">' +
      '<path class="ln dim" d="M4 80h112"/>' +
      '<path class="ln leaf" d="M22 80V54"/>' +
      '<path class="ln leaf soft" d="M22 64c-9-1-13-7-13-14 9 1 13 6 13 14zM22 59c7-1 11-6 11-13-8 1-11 6-11 13z"/>' +
      '<circle class="tomato" cx="14" cy="72" r="3.6"/>' +
      '<path class="ln leaf" d="M46 80V50"/>' +
      '<path class="ln leaf soft" d="M46 62c-9-1-13-7-13-14 9 1 13 6 13 14zM46 56c7-1 11-6 11-13-8 1-11 6-11 13z"/>' +
      '<circle class="tomato" cx="53" cy="69" r="3.6"/>' +
      '<path class="ln" d="M72 80V62"/><rect class="ln soft" x="67" y="51" width="10" height="12" rx="2.5"/>' +
      '<path class="ln acc signal" d="M81 53a7 7 0 0 1 0 8M85 49a13 13 0 0 1 0 16"/>' +
      '<path class="ln" d="M104 80V62M98 62h12"/><circle class="fillc" cx="104" cy="59" r="2.6"/>' +
      '<path class="ln spray" d="M103 57C96 38 70 30 48 36M103 57C98 44 84 40 70 42"/>' +
      '<g class="water">' +
      '<path class="drop" d="M28 30c-1.5 2.5-2.5 4-2.5 5a2.5 2.5 0 0 0 5 0c0-1-1-2.5-2.5-5z"/>' +
      '<path class="drop" d="M38 26c-1.5 2.5-2.5 4-2.5 5a2.5 2.5 0 0 0 5 0c0-1-1-2.5-2.5-5z"/>' +
      '<path class="drop" d="M50 31c-1.5 2.5-2.5 4-2.5 5a2.5 2.5 0 0 0 5 0c0-1-1-2.5-2.5-5z"/>' +
      '<path class="drop" d="M60 27c-1.5 2.5-2.5 4-2.5 5a2.5 2.5 0 0 0 5 0c0-1-1-2.5-2.5-5z"/>' +
      '</g></svg>',
    robot:
      '<svg class="art" viewBox="0 0 120 90" aria-hidden="true">' +
      '<path class="ln" d="M46 9v8"/><circle class="accf" cx="46" cy="7" r="3"/>' +
      '<rect class="ln soft" x="28" y="17" width="36" height="26" rx="9"/>' +
      '<circle class="eye" cx="39" cy="30" r="3.4"/><circle class="eye" cx="53" cy="30" r="3.4"/>' +
      '<rect class="ln soft" x="31" y="47" width="30" height="27" rx="7"/>' +
      '<path class="ln dim" d="M39 58h14M38 74v8M54 74v8"/>' +
      '<path class="ln" d="M31 54l-9 10M61 54l11 9"/>' +
      '<g class="pan"><path class="ln" d="M70 63h12"/><path class="ln soft" d="M82 61h28a14 8 0 0 1-28 0z"/></g>' +
      '<g class="steam"><path class="ln dim" d="M89 53c-3-4 3-6 0-10M97 53c-3-4 3-6 0-10M105 53c-3-4 3-6 0-10"/></g></svg>',
    empresa:
      '<svg class="art" viewBox="0 0 120 90" aria-hidden="true">' +
      '<path class="ln dim" d="M60 30v12M22 42h76M22 42v10M60 42v10M98 42v10"/>' +
      '<circle class="ln soft" cx="60" cy="18" r="12"/>' +
      '<path class="ln" d="M60 18c-2-3-4-4-6-4s-4 2-4 4 2 4 4 4 4-1 6-4 4-4 6-4 4 2 4 4-2 4-4 4-4-1-6-4z"/>' +
      '<rect class="ln soft" x="8" y="52" width="28" height="27" rx="7"/><circle class="ln" cx="22" cy="61" r="3.5"/><path class="ln" d="M15 73c1-4 4-6 7-6s6 2 7 6"/>' +
      '<rect class="ln soft" x="46" y="52" width="28" height="27" rx="7"/><circle class="ln" cx="60" cy="61" r="3.5"/><path class="ln" d="M53 73c1-4 4-6 7-6s6 2 7 6"/>' +
      '<rect class="ln soft" x="84" y="52" width="28" height="27" rx="7"/><circle class="ln" cx="98" cy="61" r="3.5"/><path class="ln" d="M91 73c1-4 4-6 7-6s6 2 7 6"/>' +
      '<circle class="accf alert" cx="110" cy="52" r="4.5"/></svg>',
    plato:
      '<svg class="art" viewBox="0 0 40 30" aria-hidden="true">' +
      '<rect class="ln soft" x="1.5" y="1.5" width="37" height="27" rx="4"/>' +
      '<path class="ln acc" d="M8 15h24a12 7 0 0 1-24 0z"/>' +
      '<path class="ln" d="M11 13c3-3 5 1 8-1.5s5 1 8-1"/></svg>',
  };

  class Cancelled extends Error {}

  const cards = {};
  const paths = {};
  const memories = [];
  let runId = 0;
  let touring = false;
  let expoTimer = null;

  const byId = (id) => CFG.scenes.find((scene) => scene.id === id);
  const scrolling = () => (calm() ? "auto" : "smooth");

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function restart(node, className) {
    node.classList.remove(className);
    void node.offsetWidth; // let the animation start again
    node.classList.add(className);
  }

  function sleep(ms, run) {
    return new Promise((resolve, reject) => {
      window.setTimeout(() => (run === runId ? resolve() : reject(new Cancelled())), ms);
    });
  }

  /* ---------- Building the stage ---------- */

  function buildCard(scene) {
    const card = el("button", "card");
    card.type = "button";
    card.dataset.scene = scene.id;
    card.setAttribute("aria-label", `${scene.label}: ${scene.action}.`);

    const art = el("span", "card-art");
    art.innerHTML = ART[scene.art] || "";
    const body = el("span", "card-body");
    body.append(el("span", "card-label", scene.label), el("span", "card-title", scene.title), el("span", "card-event", scene.event),
      el("span", "card-action", scene.action));
    const result = el("span", "card-result");
    result.setAttribute("aria-hidden", "true");
    card.append(art, body, result);

    card.addEventListener("click", () => {
      stopTour(true);
      // On a phone the brain and the panel sit above the cards: bring them into view to watch the data arrive.
      if (narrow.matches) core.scrollIntoView({ block: "start", behavior: scrolling() });
      play(scene);
    });
    return card;
  }

  function build() {
    const columns = { left: stage.querySelector(".col-left"), right: stage.querySelector(".col-right") };
    CFG.scenes.forEach((scene) => {
      const card = buildCard(scene);
      cards[scene.id] = card;
      (columns[scene.side] || columns.left).appendChild(card);
      const path = document.createElementNS(SVGNS, "path");
      path.setAttribute("class", "wire");
      path.id = `wire-${scene.id}`;
      wires.appendChild(path);
      paths[scene.id] = path;
    });
    CFG.apps.forEach((name) => appsEl.appendChild(el("li", "", name)));
  }

  /* Each wire leaves the side of its card that faces the brain and ends on the brain's ellipse. */
  function layout() {
    const box = stage.getBoundingClientRect();
    if (!box.width) return;
    wires.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    const c = core.getBoundingClientRect();
    const cx = c.left - box.left + c.width / 2;
    const cy = c.top - box.top + c.height / 2;
    const rx = c.width * 0.4;
    const ry = c.height * 0.42;

    Object.keys(cards).forEach((id) => {
      const r = cards[id].getBoundingClientRect();
      const left = r.left - box.left;
      const top = r.top - box.top;
      let ax;
      let ay;
      let sideways = true;
      if (r.right <= c.left + 4) {
        ax = left + r.width;
        ay = top + r.height / 2;
      } else if (r.left >= c.right - 4) {
        ax = left;
        ay = top + r.height / 2;
      } else {
        sideways = false;
        ax = left + r.width / 2;
        ay = top > cy ? top : top + r.height;
      }
      const dx = ax - cx;
      const dy = ay - cy;
      const k = 1 / Math.sqrt((dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)) || 0;
      const px = cx + dx * k;
      const py = cy + dy * k;
      let d;
      if (sideways) {
        const mx = (ax + px) / 2;
        d = `M${ax.toFixed(1)},${ay.toFixed(1)} C${mx.toFixed(1)},${ay.toFixed(1)} ${mx.toFixed(1)},${py.toFixed(1)} ${px.toFixed(1)},${py.toFixed(1)}`;
      } else {
        const my = (ay + py) / 2;
        d = `M${ax.toFixed(1)},${ay.toFixed(1)} C${ax.toFixed(1)},${my.toFixed(1)} ${px.toFixed(1)},${my.toFixed(1)} ${px.toFixed(1)},${py.toFixed(1)}`;
      }
      paths[id].setAttribute("d", d);
    });
  }

  /* Quiet traffic while nothing is pressed: small sparks running to the brain. */
  function spark() {
    if (calm() || document.hidden) return;
    const ids = Object.keys(paths).filter((id) => !paths[id].classList.contains("hot"));
    if (!ids.length) return;
    const id = ids[Math.floor(Math.random() * ids.length)];
    const dot = document.createElementNS(SVGNS, "circle");
    dot.setAttribute("r", "2.3");
    dot.setAttribute("class", "spark");
    const motion = document.createElementNS(SVGNS, "animateMotion");
    motion.setAttribute("dur", `${(2 + Math.random() * 0.6).toFixed(2)}s`);
    motion.setAttribute("begin", "indefinite");
    motion.setAttribute("fill", "freeze");
    const mpath = document.createElementNS(SVGNS, "mpath");
    mpath.setAttribute("href", `#wire-${id}`);
    mpath.setAttributeNS(XLINK, "xlink:href", `#wire-${id}`);
    motion.appendChild(mpath);
    dot.appendChild(motion);
    wires.appendChild(dot);
    if (typeof motion.beginElement === "function") motion.beginElement();
    window.setTimeout(() => dot.remove(), 2800);
  }

  /* ---------- The flow of one scene ---------- */

  function packetContent(text, art) {
    const frag = document.createDocumentFragment();
    if (art && ART[art]) {
      const pic = el("span", "packet-art");
      pic.innerHTML = ART[art];
      frag.appendChild(pic);
    } else {
      frag.appendChild(el("span", "packet-dot"));
    }
    frag.appendChild(el("span", "packet-text", text));
    return frag;
  }

  function travel(id, direction, content, run) {
    return new Promise((resolve, reject) => {
      const path = paths[id];
      const length = path.getTotalLength();
      const packet = el("div", direction === "out" ? "packet out" : "packet");
      packet.appendChild(content);
      stage.appendChild(packet);
      path.classList.add("hot");
      path.classList.toggle("out", direction === "out");
      const width = packet.offsetWidth;
      const height = packet.offsetHeight;
      const duration = 1500;
      const start = performance.now();
      const place = (t) => {
        const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        const point = path.getPointAtLength((direction === "in" ? eased : 1 - eased) * length);
        packet.style.transform = `translate(${(point.x - width / 2).toFixed(1)}px, ${(point.y - height / 2).toFixed(1)}px)`;
      };
      place(0);
      let over = false;
      let guard = 0;
      const frame = (now) => {
        if (over) return;
        if (run !== runId) {
          over = true;
          window.clearTimeout(guard);
          packet.remove();
          reject(new Cancelled());
          return;
        }
        const t = Math.min(1, (now - start) / duration);
        place(t);
        if (t < 1) {
          window.requestAnimationFrame(frame);
          return;
        }
        over = true;
        window.clearTimeout(guard);
        path.classList.remove("hot", "out");
        packet.classList.add("arrive");
        window.setTimeout(() => packet.remove(), 260);
        resolve();
      };
      // A hidden tab gets no animation frames: the journey still ends on time, so a tour never stalls.
      guard = window.setTimeout(() => frame(start + duration), duration + 300);
      window.requestAnimationFrame(frame);
    });
  }

  function pulseCore() {
    restart(core, "pulse");
    const ring = el("span", "ripple");
    ripples.appendChild(ring);
    window.setTimeout(() => ring.remove(), 1300);
  }

  function updateCount() {
    const n = memories.length;
    countEl.textContent = `${n} ${n === 1 ? "recuerdo" : "recuerdos"}`;
    restart(countEl, "bump");
  }

  function remember(scene, text) {
    const known = memories.find((m) => m.text === text);
    if (known) {
      restart(known.node, "recalled");
      return known;
    }
    const empty = memlist.querySelector(".mem-empty");
    if (empty) empty.remove();
    const item = el("li");
    item.append(el("span", "mem-dot"), el("span", "mem-text", text), el("span", "mem-src", `· ${scene.label}`));
    memlist.appendChild(item);
    memlist.scrollLeft = memlist.scrollWidth;
    const memory = { scene: scene.id, text, node: item };
    memories.push(memory);
    updateCount();
    return memory;
  }

  function recallFrom(sceneId) {
    const source = byId(sceneId);
    if (!source || !source.memory) return;
    const memory = memories.find((m) => m.scene === sceneId) || remember(source, source.memory);
    restart(memory.node, "recalled");
    memlist.scrollTo({ left: memory.node.offsetLeft - memlist.offsetLeft - 8, behavior: scrolling() });
  }

  async function lightApps(run) {
    for (const chip of appsEl.children) {
      chip.classList.add("lit");
      await sleep(150, run);
    }
    await sleep(500, run);
  }

  function renderSteps(scene) {
    panel.steps.textContent = "";
    CFG.steps.forEach((step) => {
      const item = el("li", "", step.label);
      item.dataset.step = step.id;
      const applies = !scene || step.id === "recoge" || step.id === "recuerda" ||
        (step.id === "analiza" && scene.think && scene.think.length > 0) || (step.id === "actua" && scene.output);
      if (!applies) item.classList.add("skip");
      panel.steps.appendChild(item);
    });
  }

  function setStep(id) {
    let reached = false;
    for (const item of panel.steps.children) {
      if (item.dataset.step === id) {
        reached = true;
        item.classList.remove("done");
        item.classList.add("on");
      } else if (!reached) {
        item.classList.remove("on");
        if (!item.classList.contains("skip")) item.classList.add("done");
      }
    }
  }

  function finishSteps() {
    for (const item of panel.steps.children) {
      item.classList.remove("on");
      if (!item.classList.contains("skip")) item.classList.add("done");
    }
  }

  function showIdle() {
    panel.title.textContent = CFG.idle.title;
    panel.text.textContent = CFG.idle.text;
    panel.analysis.textContent = "";
    panel.result.hidden = true;
    renderSteps(null);
  }

  function showScene(scene) {
    panel.title.textContent = `${scene.label} · ${scene.action}`;
    panel.text.textContent = scene.explain;
    panel.analysis.textContent = "";
    panel.result.hidden = true;
    renderSteps(scene);
  }

  function clearStage() {
    stage.querySelectorAll(".packet").forEach((node) => node.remove());
    Object.values(paths).forEach((path) => path.classList.remove("hot", "out"));
    Object.values(cards).forEach((card) => card.classList.remove("active", "emit"));
    for (const chip of appsEl.children) chip.classList.remove("lit");
    core.classList.remove("thinking", "pulse");
  }

  function finish(card, text) {
    card.querySelector(".card-result").textContent = `✓ ${text}`;
    card.classList.add("done");
  }

  async function play(scene) {
    const run = ++runId;
    clearStage();
    const card = cards[scene.id];
    card.classList.remove("done", "acting");
    card.classList.add("active");
    showScene(scene);
    try {
      setStep("recoge");
      restart(card, "emit");
      await sleep(800, run);
      await travel(scene.id, "in", packetContent(scene.packet, scene.packetArt), run);

      setStep("recuerda");
      pulseCore();
      if (scene.memory) remember(scene, scene.memory);
      if (scene.recalls) recallFrom(scene.recalls);
      await sleep(900, run);

      if (scene.think && scene.think.length) {
        setStep("analiza");
        core.classList.add("thinking");
        for (const line of scene.think) {
          panel.analysis.appendChild(el("li", "", line));
          await sleep(900, run);
        }
        core.classList.remove("thinking");
      }

      const output = scene.output;
      let resultText = "Guardado en la memoria común";
      if (output) {
        setStep("actua");
        resultText = output.result;
        if (output.to === "apps") {
          await lightApps(run);
          finish(card, output.result);
        } else {
          const target = output.to === "self" ? scene.id : output.to;
          await travel(target, "out", packetContent(output.packet), run);
          const targetCard = cards[target];
          restart(targetCard, "acting");
          window.setTimeout(() => targetCard.classList.remove("acting"), 7000);
          finish(targetCard, output.result);
        }
      } else {
        finish(card, resultText);
      }
      panel.result.textContent = resultText;
      panel.result.hidden = false;
      finishSteps();
      return true;
    } catch (error) {
      if (error instanceof Cancelled) return false;
      throw error;
    }
  }

  function resetAll() {
    runId += 1;
    clearStage();
    memories.length = 0;
    memlist.textContent = "";
    memlist.appendChild(el("li", "mem-empty", "Todavía vacía: pulsa una escena."));
    countEl.textContent = "0 recuerdos";
    Object.values(cards).forEach((card) => {
      card.classList.remove("done", "acting");
      card.querySelector(".card-result").textContent = "";
    });
    showIdle();
  }

  /* ---------- Guided tour ---------- */

  function setTourButton(on) {
    tourBtn.setAttribute("aria-pressed", on ? "true" : "false");
    tourBtn.querySelector("span").textContent = on ? "Pausar recorrido" : "Ver recorrido";
  }

  function stopTour(byPerson) {
    if (touring) {
      touring = false;
      setTourButton(false);
    }
    if (expo && byPerson) {
      window.clearTimeout(expoTimer);
      expoTimer = window.setTimeout(() => {
        if (!touring) tour();
      }, 45000);
    }
  }

  async function tour() {
    touring = true;
    setTourButton(true);
    try {
      do {
        resetAll();
        for (const scene of CFG.scenes) {
          if (!touring) return;
          const completed = await play(scene);
          if (!completed || !touring) return;
          await sleep(2600, runId);
        }
        if (expo && touring) await sleep(5000, runId);
      } while (expo && touring);
    } catch (error) {
      if (!(error instanceof Cancelled)) throw error;
    } finally {
      if (touring) {
        touring = false;
        setTourButton(false);
      }
    }
  }

  tourBtn.addEventListener("click", () => {
    if (touring) {
      stopTour(true);
      runId += 1;
      clearStage();
      return;
    }
    tour();
    stage.scrollIntoView({ block: "nearest", behavior: scrolling() });
  });

  document.getElementById("reset").addEventListener("click", () => {
    stopTour(true);
    resetAll();
  });

  /* ---------- Demo dialog ---------- */

  const dialog = document.getElementById("demo");
  document.querySelectorAll("[data-open-demo]").forEach((button) => {
    button.addEventListener("click", () => {
      if (dialog && typeof dialog.showModal === "function") dialog.showModal();
      else window.location.href = "https://infinitomemory.com/";
    });
  });
  if (dialog) {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  }

  /* ---------- Starry background ---------- */

  function stars() {
    const canvas = document.querySelector(".stars");
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    let width = 0;
    let height = 0;
    let points = [];

    function size() {
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const n = Math.round(Math.min(90, (width * height) / 16000));
      points = Array.from({ length: n }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.14,
        vy: (Math.random() - 0.5) * 0.14,
        r: Math.random() * 1.3 + 0.3,
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      for (let i = 0; i < points.length; i += 1) {
        const a = points[i];
        for (let j = i + 1; j < points.length; j += 1) {
          const b = points[j];
          const dist = Math.hypot(a.x - b.x, a.y - b.y);
          if (dist < 120) {
            ctx.strokeStyle = `rgba(103, 232, 249, ${((1 - dist / 120) * 0.12).toFixed(3)})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      ctx.fillStyle = "rgba(186, 240, 255, .55)";
      points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    let last = 0;
    function tick(now) {
      window.requestAnimationFrame(tick);
      if (document.hidden || now - last < 33) return;
      last = now;
      points.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -10) p.x = width + 10;
        if (p.x > width + 10) p.x = -10;
        if (p.y < -10) p.y = height + 10;
        if (p.y > height + 10) p.y = -10;
      });
      draw();
    }

    size();
    let resizeTimer = null;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        size();
        draw();
      }, 200);
    });
    if (calm()) draw();
    else window.requestAnimationFrame(tick);
  }

  /* ---------- Start ---------- */

  build();
  showIdle();
  layout();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(layout);
    [stage, core, ...Object.values(cards)].forEach((node) => observer.observe(node));
  }
  window.addEventListener("resize", layout);
  window.addEventListener("load", layout);
  window.setInterval(spark, 1200);
  stars();
  if (expo) window.setTimeout(() => tour(), 1500);
})();

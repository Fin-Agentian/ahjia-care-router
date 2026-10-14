/* AhJia Care Router Workbench. Every value shown comes from the embedded replay data (DATA).
   Timing and power are simulated by the Hailo-10H simulator; the page says so wherever they appear. */
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("ahjia-data").textContent);
  var META = DATA.meta;
  var NS = "http://www.w3.org/2000/svg";
  var EPS = 1e-6;
  var DAY = 86400;
  var SPEECH_SHOW_S = 4.0;
  var CONFIRM_SHOW_S = 4.0;
  var R30 = META.rules.escalate_no_answer_s;
  var STAND_S = META.rules.standing_clear_s;
  var FULL = META.full_fps;
  var BY_NAME = {};
  DATA.scenarios.forEach(function (s) { BY_NAME[s.name] = s; });

  var ACTION_COLOR = { A0: "#5B6B70", A1: "#8FB9A8", A2: "#2E7D74", A3: "#3A6EA5", A4: "#F2A541", A5: "#C8553D" };
  var ACTION_SHORT = { A0: "log only", A1: "template voice", A2: "wake LLM", A3: "look closer", A4: "family text", A5: "call family" };
  var RESP_NAME = { A1: "voice", A2: "LLM reply", A4: "family text", A5: "call" };
  var WHO_COLOR = { elder: "#F2A541", family: "#8FB9A8", caregiver: "#5E93CF", stranger: "#C8553D", unknown: "#A9BCBA" };
  var WHO_NAME = { elder: "Elder", family: "Family", caregiver: "Caregiver", stranger: "Stranger", unknown: "Person" };
  var POSE_NAME = { stand: "standing", sit: "sitting", lie: "lying", unknown: "" };
  var CAM_STATE = {
    empty: { color: "#7A8B90", label: "empty", alpha: 0.07 },
    occ: { color: "#3FA396", label: "person", alpha: 0.16 },
    high: { color: "#F2A541", label: "high risk", alpha: 0.2 },
    look: { color: "#5E93CF", label: "A3 look closer", alpha: 0.3 }
  };
  var SERIES = { S0: "#7D8F94", S1: "#3A6EA5", S2: "#39A193" };
  var ALERT_ACTIVE = { pre_alert: 1, escalated: 1 };

  /* ---------------------------------------------------------------- helpers */
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent, text) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined && text !== null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function sv(tag, attrs, parent, text) {
    var n = document.createElementNS(NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined && text !== null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function parseClock(c) { var p = c.split(":").map(Number); return p[0] * 3600 + p[1] * 60 + (p[2] || 0); }
  function fmtClock(s) { s = ((Math.floor(s + EPS) % DAY) + DAY) % DAY; return pad(Math.floor(s / 3600)) + ":" + pad(Math.floor(s / 60) % 60) + ":" + pad(s % 60); }
  function num(v, d) { return v === null || v === undefined || v === "" ? null : Number(v).toFixed(d); }
  function fmtFps(f) { return f === null || f === undefined ? "–" : (Math.abs(f - Math.round(f)) < 1e-9 ? String(Math.round(f)) : f.toFixed(1)); }
  function fmtSec(v) { return String(Number(Number(v).toFixed(1))); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function textWidth(s, size) {
    var w = 0;
    for (var i = 0; i < s.length; i++) w += s.charCodeAt(i) > 0x2e80 ? size : size * 0.52;
    return w;
  }
  function wrap(s, maxW, size) {
    var words = s.split(/(\s+)/), lines = [], cur = "";
    words.forEach(function (w) {
      if (textWidth(cur + w, size) > maxW && cur.trim()) { lines.push(cur.trim()); cur = w.trim(); } else cur += w;
    });
    if (cur.trim()) lines.push(cur.trim());
    return lines.slice(0, 3);
  }
  function lastAt(changes, t) {
    var lo = 0, hi = changes.length;
    while (lo < hi) { var m = (lo + hi) >> 1; if (changes[m][0] <= t + 1e-9) lo = m + 1; else hi = m; }
    return lo ? changes[lo - 1][1] : [];
  }

  /* ---------------------------------------------------------------- state */
  var Q = new URLSearchParams(location.search);
  var S = {
    scn: BY_NAME[Q.get("scenario")] || BY_NAME.night_fall || DATA.scenarios[0],
    strategy: ["S0", "S1", "S2"].indexOf(Q.get("strategy")) >= 0 ? Q.get("strategy") : "S2",
    tab: ["live", "compare", "how"].indexOf(Q.get("tab")) >= 0 ? Q.get("tab") : "live",
    t: 0, playing: false, speed: "1", record: Q.get("record") === "1", last: null
  };
  // Future chips and response markers: hidden until their time when recording, dimmed otherwise.
  S.future = ["hide", "dim"].indexOf(Q.get("future")) >= 0 ? Q.get("future") : (S.record ? "hide" : "dim");
  var FUT = [];
  function run() { return S.scn.runs[S.strategy]; }
  function startS() { return parseClock(S.scn.start_clock); }
  function clockAt(t) { return fmtClock(startS() + t); }
  function isNight(t) {
    var s = (startS() + Math.floor(t + EPS)) % DAY, a = parseClock(META.night[0]), b = parseClock(META.night[1]);
    return a > b ? (s >= a || s < b) : (s >= a && s < b);
  }
  function speedFactor() {
    if (S.speed === "fast") return S.scn.duration_s > 3600 ? 600 : 8;
    return Number(S.speed);
  }

  /* ---------------------------------------------------------------- data at time t */
  function worldAt(t) {
    var w = S.scn.world, cams = {};
    Object.keys(w.cams).forEach(function (c) { cams[c] = lastAt(w.cams[c], t); });
    var show = SPEECH_SHOW_S * Math.max(1, speedFactor() / 4);
    var speech = w.speech.filter(function (s) { return s.t - 1e-9 <= t && t < s.t + show; });
    var confirm = w.app_confirm.some(function (c) { return c - 1e-9 <= t && t < c + CONFIRM_SHOW_S; });
    return { cams: cams, speech: speech, app_confirm: confirm };
  }
  function snapAt(t) {
    var tl = run().timeline;
    if (!tl || !tl.length) return null;
    var lo = 0, hi = tl.length;
    while (lo < hi) { var m = (lo + hi) >> 1; if (tl[m].t <= t + 1e-9) lo = m + 1; else hi = m; }
    return tl[Math.max(0, lo - 1)];
  }
  function decisionsUntil(t) { return run().decisions.filter(function (d) { return d.t <= t + 1e-9; }); }
  function alertInfo(t) {
    var st = "none", start = null, setBy = null, a3 = false, last = null;
    decisionsUntil(t).forEach(function (d) {
      last = d;
      if (d.alert !== st) {
        if (ALERT_ACTIVE[d.alert] && !ALERT_ACTIVE[st]) { start = d.t; a3 = false; }
        st = d.alert; setBy = d;
      }
      if (d.actions.indexOf("A3") >= 0) a3 = true;
    });
    if (!ALERT_ACTIVE[st]) a3 = false;
    return { state: st, start: start, setBy: setBy, a3: a3, last: last };
  }
  function fpsAt(t) {
    var snap = snapAt(t);
    if (snap) return snap.fps;
    var ds = decisionsUntil(t);
    return ds.length ? ds[ds.length - 1].fps : {};
  }
  function elderPoseTimeline() {
    var out = [];
    Object.keys(S.scn.world.cams).forEach(function (cam) {
      S.scn.world.cams[cam].forEach(function (ch) {
        var e = ch[1].filter(function (p) { return p.who === "elder"; })[0];
        if (e) out.push([ch[0], e.pose]);
      });
    });
    return out.sort(function (a, b) { return a[0] - b[0]; });
  }

  /* ---------------------------------------------------------------- plan geometry (SVG units) */
  var ZONE_AT = {
    bed: [118, 122], bedside: [231, 140], sofa: [165, 446], living: [470, 408], kitchen: [880, 455],
    bathroom: [480, 140], bathroom_door: [480, 262], hallway: [250, 262], front_door: [926, 140]
  };
  var FLOOR_AT = { hallway: [345, 264], living: [620, 425], bedroom: [292, 188], bathroom_door: [520, 264], front_door: [862, 245] };
  var CAMS = {
    bedroom: { x: 26, y: 48, aim: 36, fov: 78, range: 300, badge: [26, 2], name: "bedroom", clip: [14, 40, 346, 175] },
    hallway: { x: 20, y: 255, aim: 0, fov: 30, range: 760, badge: [36, 268], name: "hallway", clip: [14, 215, 766, 80] },
    bathroom_door: { x: 548, y: 291, aim: 228, fov: 56, range: 130, badge: [560, 222], name: "bathroom door", clip: [14, 150, 766, 145] },
    front_door: { x: 788, y: 46, aim: 28, fov: 72, range: 240, badge: [794, 2], name: "front door", clip: [780, 40, 206, 255] },
    living: { x: 980, y: 302, aim: 160, fov: 72, range: 720, badge: [790, 302], name: "living", clip: [14, 295, 972, 195] }
  };
  var TV_AT = [170, 312], BOX_AT = [273, 318];

  function drawPlanStatic(svg) {
    clear(svg);
    var g = sv("g", { "class": "static" }, svg);
    function room(x, y, w, h, alt) { sv("rect", { x: x, y: y, width: w, height: h, "class": "room" + (alt ? " alt" : "") }, g); }
    function label(x, y, t, anchor) { sv("text", { x: x, y: y, "class": "room-label", "text-anchor": anchor || "start" }, g, t); }
    function furn(x, y, w, h, rx, cls) { sv("rect", { x: x, y: y, width: w, height: h, rx: rx || 4, "class": cls || "furn" }, g); }
    function gap(x1, y1, x2, y2) { sv("line", { x1: x1, y1: y1, x2: x2, y2: y2, "class": "door-gap" }, g); }
    function swing(d) { sv("path", { d: d, "class": "door-swing" }, g); }
    room(14, 40, 346, 175); room(360, 40, 240, 175, 1); room(600, 40, 180, 175, 1); room(780, 40, 206, 255, 1);
    room(14, 215, 766, 80, 1); room(14, 295, 972, 195);
    // high-risk zones (DESIGN: bathroom door, bedside; hallway at night)
    sv("rect", { x: 196, y: 82, width: 70, height: 96, rx: 6, "class": "zone-hi" }, g);
    sv("text", { x: 231, y: 76, "class": "zone-label", "text-anchor": "middle" }, g, "bedside");
    sv("rect", { x: 428, y: 220, width: 104, height: 70, rx: 6, "class": "zone-hi" }, g);
    // doors
    gap(292, 215, 344, 215); swing("M292 215 A52 52 0 0 1 344 163");
    gap(445, 215, 515, 215); swing("M445 215 A70 70 0 0 0 515 145");
    gap(650, 215, 700, 215); swing("M650 215 A50 50 0 0 1 700 165");
    gap(780, 225, 780, 287);
    gap(560, 295, 700, 295);
    gap(986, 92, 986, 168); swing("M986 92 A76 76 0 0 0 910 168");
    sv("text", { x: 975, y: 186, "class": "zone-label", "text-anchor": "end" }, g, "front door");
    // bedroom
    furn(40, 68, 150, 108, 8); furn(44, 74, 34, 96, 6, "furn2"); furn(200, 90, 40, 40, 4);
    furn(250, 46, 96, 26, 3);
    // bathroom
    furn(372, 50, 112, 56, 18); furn(548, 52, 40, 34, 10); furn(560, 150, 30, 42, 12);
    // study
    furn(612, 52, 120, 40, 4); furn(650, 98, 40, 30, 12, "furn2");
    // entry
    furn(800, 230, 30, 54, 3);
    // living
    furn(44, 424, 236, 52, 14); furn(44, 408, 236, 22, 10, "furn2");
    furn(110, 360, 116, 38, 8, "furn2");
    furn(112, 299, 116, 9, 2, "furn2");
    sv("rect", { x: 251, y: 300, width: 44, height: 14, rx: 4, fill: "#0B1E24", stroke: "#3FA396", "stroke-width": 1.5 }, g);
    sv("circle", { cx: 259, cy: 307, r: 2.6, fill: "#3FA396" }, g);
    sv("text", { x: 273, y: 330, "text-anchor": "middle", "class": "zone-label", fill: "#7fd0c4", style: "fill:#7fd0c4" }, g, "AhJia box");
    furn(824, 404, 156, 28, 3); furn(952, 300, 28, 104, 3); furn(860, 410, 24, 16, 8, "furn2"); furn(900, 410, 24, 16, 8, "furn2");
    label(26, 205, "Bedroom"); label(588, 205, "Bathroom", "end"); label(768, 205, "Study", "end");
    label(974, 62, "Entry", "end"); label(770, 287, "Hallway", "end"); label(300, 482, "Living room");
    label(974, 482, "Kitchen", "end");
    var defs = sv("defs", null, svg);
    Object.keys(CAMS).forEach(function (k) {
      var c = CAMS[k].clip, cp = sv("clipPath", { id: "clip-" + k }, defs);
      sv("rect", { x: c[0], y: c[1], width: c[2], height: c[3] }, cp);
    });
    sv("g", { "class": "wedges" }, svg);
    sv("g", { "class": "dyn" }, svg);
  }

  function wedgePath(c) {
    var a1 = (c.aim - c.fov / 2) * Math.PI / 180, a2 = (c.aim + c.fov / 2) * Math.PI / 180;
    var x1 = c.x + c.range * Math.cos(a1), y1 = c.y + c.range * Math.sin(a1);
    var x2 = c.x + c.range * Math.cos(a2), y2 = c.y + c.range * Math.sin(a2);
    return "M" + c.x + " " + c.y + " L" + x1.toFixed(1) + " " + y1.toFixed(1) + " A" + c.range + " " + c.range +
      " 0 0 1 " + x2.toFixed(1) + " " + y2.toFixed(1) + " Z";
  }

  function camState(cam, persons, night, fps, info) {
    var risky = META.high_risk_zones.concat(night ? META.night_high_risk_zones : []);
    if (info.a3 && fps !== undefined && fps >= FULL - 1e-6 && persons.length) return "look";
    if (!persons.length) return "empty";
    var lying = persons.some(function (p) { return p.pose === "lie" && META.rest_zones.indexOf(p.zone) < 0; });
    if (risky.indexOf(cam) >= 0 || lying || persons.some(function (p) { return risky.indexOf(p.zone) >= 0; })) return "high";
    return "occ";
  }

  function personGlyph(g, pose, color) {
    var d;
    if (pose === "lie") d = ["M-15 -2 L12 -2", "M12 -2 L27 1 M12 -2 L27 -6", "M-6 -2 L2 7"];
    else if (pose === "sit") d = ["M-3 -21 L-1 -3", "M-1 -3 L13 -3 L13 11", "M-2 -15 L9 -10"];
    else d = ["M0 -24 L0 -3", "M0 -3 L-7 13 M0 -3 L7 13", "M-9 -15 L0 -19 L9 -15"];
    var head = pose === "lie" ? [-23, -3] : (pose === "sit" ? [-4, -29] : [0, -32]);
    d.forEach(function (p) { sv("path", { d: p, stroke: "#0B1E24", "stroke-width": 9, "stroke-linecap": "round", "stroke-linejoin": "round", fill: "none" }, g); });
    d.forEach(function (p) { sv("path", { d: p, stroke: color, "stroke-width": 5, "stroke-linecap": "round", "stroke-linejoin": "round", fill: "none" }, g); });
    sv("circle", { cx: head[0], cy: head[1], r: 8, fill: color, stroke: "#0B1E24", "stroke-width": 2 }, g);
  }

  function drawBubble(g, sp, at) {
    var size = 15, isTv = sp.source === "tv";
    var lines = wrap("“" + sp.text + "”", 250, size);
    var w = Math.max(90, Math.max.apply(null, lines.map(function (l) { return textWidth(l, size); })) + 28 + (isTv ? 26 : 0));
    var h = 24 + lines.length * 19;
    var x = clamp(at[0] - w / 2, 18, 982 - w), y = at[1] - 58 - h, below = false;
    if (y < 44) { y = at[1] + 30; below = true; }
    var b = sv("g", { "class": "bubble" + (isTv ? " tv" : "") }, g);
    var tx = clamp(at[0], x + 14, x + w - 14);
    sv("rect", { x: x, y: y, width: w, height: h, rx: 12 }, b);
    sv("path", { d: below ? ("M" + (tx - 8) + " " + y + " L" + tx + " " + (y - 12) + " L" + (tx + 8) + " " + y) :
      ("M" + (tx - 8) + " " + (y + h) + " L" + tx + " " + (y + h + 12) + " L" + (tx + 8) + " " + (y + h)),
      fill: isTv ? "#FFE7C2" : "#F7FAF9", stroke: "#0B1E24", "stroke-width": 1 }, b);
    var who = (isTv ? "TV" : (WHO_NAME[sp.source] || sp.source)) + (sp.directed ? "  → to AhJia" : "");
    var ox = x + 14;
    if (isTv) {
      sv("rect", { x: ox, y: y + 7, width: 18, height: 12, rx: 2, fill: "#5B6B70" }, b);
      sv("path", { d: "M" + (ox + 5) + " " + (y + 22) + " L" + (ox + 13) + " " + (y + 22), stroke: "#5B6B70", "stroke-width": 2 }, b);
      ox += 26;
    }
    sv("text", { x: ox, y: y + 17, "class": "who" }, b, who);
    lines.forEach(function (l, i) { sv("text", { x: x + 14 + (isTv ? 26 : 0), y: y + 37 + i * 19 }, b, l); });
  }

  function placePeople(world) {
    var placed = [], seen = {}, used = {};
    S.scn.cameras.forEach(function (cam) {
      (world.cams[cam] || []).forEach(function (p) {
        if (seen[p.id]) return;
        seen[p.id] = 1;
        var at = (p.zone !== "floor" && p.zone !== "other" && ZONE_AT[p.zone]) || FLOOR_AT[cam] || [500, 260];
        var key = at.join(","), n = used[key] || 0;
        used[key] = n + 1;
        placed.push({ p: p, cam: cam, x: at[0] + n * 58, y: at[1] });
      });
    });
    return placed;
  }

  function drawPlanDyn(t) {
    var svg = $("plan"), wedges = svg.querySelector(".wedges"), g = svg.querySelector(".dyn");
    clear(wedges); clear(g);
    var world = worldAt(t), fps = fpsAt(t), info = alertInfo(t), night = isNight(t);
    var people = placePeople(world);
    S.scn.cameras.forEach(function (cam) {
      var c = CAMS[cam];
      if (!c) return;
      var st = CAM_STATE[camState(cam, world.cams[cam] || [], night, fps[cam], info)];
      sv("path", { d: wedgePath(c), "class": "wedge", fill: st.color, "fill-opacity": st.alpha, stroke: st.color, "stroke-opacity": 0.55, "clip-path": "url(#clip-" + cam + ")" }, wedges);
    });
    S.scn.cameras.forEach(function (cam) {
      var c = CAMS[cam];
      if (!c) return;
      var key = camState(cam, world.cams[cam] || [], night, fps[cam], info), st = CAM_STATE[key];
      var cg = sv("g", { transform: "translate(" + c.x + "," + c.y + ") rotate(" + c.aim + ")" }, g);
      sv("rect", { x: -9, y: -7, width: 18, height: 14, rx: 3, fill: st.color, "class": "cam-body" }, cg);
      sv("path", { d: "M9 -4 L15 -7 L15 7 L9 4 Z", fill: st.color, "class": "cam-body" }, cg);
      var fpsText = fmtFps(fps[cam]) + " fps";
      var bx = c.badge[0], by = c.badge[1], bw = 36 + (c.name.length + fpsText.length) * 7.4, bh = 36;
      var bg = sv("g", { "class": "fps-badge" }, g);
      sv("rect", { x: bx, y: by, width: bw, height: bh, rx: 8, stroke: st.color }, bg);
      sv("circle", { cx: bx + 12, cy: by + 13, r: 4.5, fill: st.color }, bg);
      sv("text", { x: bx + 22, y: by + 17 }, bg, c.name);
      sv("text", { x: bx + bw - 9, y: by + 17, "text-anchor": "end", style: "font-weight:700" }, bg, fpsText);
      sv("text", { x: bx + 10, y: by + 30, "class": "st", fill: st.color, style: "fill:" + st.color }, bg, st.label.toUpperCase());
    });
    people.forEach(function (it) {
      var p = it.p, color = WHO_COLOR[p.who] || WHO_COLOR.unknown;
      var pg = sv("g", { transform: "translate(" + it.x + "," + it.y + ")" }, g);
      if (p.pose === "lie" && META.rest_zones.indexOf(p.zone) < 0) {
        sv("circle", { cx: 0, cy: -2, r: 34, "class": "ring" }, pg);
        sv("circle", { cx: 0, cy: -2, r: 34, "class": "ring pulse" }, pg);
      }
      personGlyph(pg, p.pose, color);
      sv("text", { x: 0, y: 30, "text-anchor": "middle", "class": "person-label", fill: color }, pg,
        (WHO_NAME[p.who] || p.who) + (POSE_NAME[p.pose] ? " · " + POSE_NAME[p.pose] : ""));
    });
    world.speech.forEach(function (sp) {
      var at = BOX_AT;
      if (sp.source === "tv") at = TV_AT;
      else {
        var who = people.filter(function (it) { return it.p.who === sp.source; })[0];
        if (who) at = [who.x, who.y - 34];
      }
      drawBubble(g, sp, at);
    });
    if (world.app_confirm || info.state === "cleared_by_app") {
      var oc = sv("g", { "class": "overlay-chip" }, g);
      sv("rect", { x: 380, y: 4, width: 290, height: 32, rx: 9 }, oc);
      sv("rect", { x: 392, y: 10, width: 12, height: 20, rx: 3, fill: "none", stroke: "#8FB9A8", "stroke-width": 2 }, oc);
      sv("text", { x: 414, y: 25 }, oc, "Family confirmed in the app");
    }
  }

  /* ---------------------------------------------------------------- timeline */
  var TL = { w: 1206, h: 244, left: 150, right: 18, truthY: 4, laneY: 46, laneH: 18, laneGap: 2, respY: 172, axisY: 226 };
  var LANES = ["A5", "A4", "A3", "A2", "A1", "A0"];
  function tx(t) { return TL.left + (t / S.scn.duration_s) * (TL.w - TL.left - TL.right); }

  function greedyRows(items, rows, gapPx) {
    var ends = []; for (var i = 0; i < rows; i++) ends.push(-1e9);
    return items.map(function (it) {
      for (var r = 0; r < rows; r++) if (it.x0 > ends[r] + gapPx) { ends[r] = it.x1; return r; }
      return -1;
    });
  }

  function chipTip(d) {
    var lat = d.latency_s === null || d.latency_s === undefined ? "no response time (log only)" : "latency " + d.latency_s.toFixed(2) + " s (simulated)";
    return d.clock + " · " + (d.agent || "-") + " · " + d.actions.join("+") + "\n" + d.reason + "\n" + lat;
  }

  function drawTimelineLanes(g) {
    sv("rect", { x: TL.left, y: TL.truthY, width: TL.w - TL.left - TL.right, height: 38, "class": "lane-bg" }, g);
    sv("text", { x: 8, y: TL.truthY + 16, "class": "lane-label" }, g, "What happens");
    sv("text", { x: 8, y: TL.truthY + 31, "class": "lane-label", style: "font-size:11px;fill:#7F9693" }, g, "scenario file");
    LANES.forEach(function (a, i) {
      var y = TL.laneY + i * (TL.laneH + TL.laneGap);
      sv("rect", { x: TL.left, y: y, width: TL.w - TL.left - TL.right, height: TL.laneH, "class": "lane-bg" }, g);
      sv("rect", { x: 8, y: y + 2, width: 30, height: TL.laneH - 4, rx: 4, fill: ACTION_COLOR[a] }, g);
      sv("text", { x: 23, y: y + 13.5, "text-anchor": "middle", "class": "lane-code", fill: a === "A1" || a === "A4" ? "#0B1E24" : "#fff" }, g, a);
      sv("text", { x: 46, y: y + 13.5, "class": "lane-label" }, g, ACTION_SHORT[a]);
    });
    sv("rect", { x: TL.left, y: TL.respY, width: TL.w - TL.left - TL.right, height: 50, "class": "lane-bg" }, g);
    sv("text", { x: 8, y: TL.respY + 20, "class": "lane-label" }, g, "Responses");
    sv("text", { x: 8, y: TL.respY + 36, "class": "lane-label", style: "fill:#F2A541;font-size:11.5px" }, g, "simulated timing");
  }

  function drawTimelineAxis(g, dur) {
    var ax = sv("g", { "class": "axis" }, g);
    var step = dur > 20000 ? 7200 : dur > 300 ? 60 : dur > 100 ? 20 : 10;
    for (var s = 0; s <= dur + EPS; s += step) {
      var x = tx(s);
      sv("line", { x1: x, y1: TL.truthY, x2: x, y2: TL.axisY, "stroke-dasharray": "2 4" }, ax);
      var lab = dur > 20000 ? clockAt(s).slice(0, 5) : clockAt(s);
      sv("text", { x: x, y: TL.axisY + 13, "text-anchor": s === 0 ? "start" : (s + step > dur + EPS ? "end" : "middle") }, ax, lab);
    }
  }

  // what happens in the scenario file: labels (+ speech, app confirms and the elder's posture changes for short scenarios)
  function truthItems(short) {
    var truth = S.scn.labels.map(function (lb) { return { t: lb.t, text: "▲ " + lb.what, kind: lb.kind, label: true }; });
    if (short) S.scn.world.speech.forEach(function (sp) {
      var txt = sp.text.length > 26 ? sp.text.slice(0, 25) + "…" : sp.text;
      truth.push({ t: sp.t, text: (sp.source === "tv" ? "TV: " : "") + "“" + txt + "”" });
    });
    if (short) S.scn.world.app_confirm.forEach(function (c) { truth.push({ t: c, text: "\u2713 app confirm" }); });
    if (short) {
      var prev = null, verbs = { stand: "\u2191 stands", sit: "sits", lie: "\u2193 lies down" };
      elderPoseTimeline().forEach(function (p) {
        if (prev !== null && p[1] !== prev && verbs[p[1]]) truth.push({ t: p[0], text: verbs[p[1]] });
        prev = p[1];
      });
    }
    truth.sort(function (a, b) { return a.t - b.t; });
    return truth.filter(function (it, i) { return !it.label || !truth.slice(0, i).some(function (o) { return o.label && o.text === it.text; }); });
  }

  function drawTruthLane(g, short) {
    var truth = truthItems(short);
    var items = truth.map(function (it) { var x = tx(it.t); return { x0: x - 4, x1: x + textWidth(it.text, 11.5) + 4 }; });
    var rows = greedyRows(items, 2, 6);
    truth.forEach(function (it, i) {
      if (rows[i] < 0) return;
      var x = tx(it.t), y = TL.truthY + 15 + rows[i] * 17;
      sv("text", { x: x - 4, y: y, "class": "truth-l", style: it.label ? "" : "fill:#CFE0DD" }, g, it.text);
    });
  }

  function hoverTip(node, text) {
    node.addEventListener("mouseenter", function (ev) { showTip(ev, text); });
    node.addEventListener("mousemove", moveTip);
    node.addEventListener("mouseleave", hideTip);
  }

  function drawDecisionChips(g, short) {
    var chips = sv("g", null, g);
    FUT = [];
    run().decisions.forEach(function (d) {
      d.actions.forEach(function (a) {
        var cg = sv("g", null, chips);
        FUT.push({ el: cg, t: d.t });
        var lane = LANES.indexOf(a);
        if (lane < 0) return;
        var y = TL.laneY + lane * (TL.laneH + TL.laneGap) + 2, x = tx(d.t), w = short ? 24 : 4;
        var r = sv("rect", { x: x - 2, y: y, width: w, height: TL.laneH - 4, rx: short ? 4 : 1, fill: ACTION_COLOR[a], "class": "chip-r" }, cg);
        hoverTip(r, chipTip(d));
        if (short) sv("text", { x: x + 10, y: y + 10.5, "text-anchor": "middle", "class": "chip-t", fill: a === "A1" || a === "A4" ? "#0B1E24" : "#fff", style: a === "A1" || a === "A4" ? "" : "fill:#fff" }, cg, a);
      });
    });
  }

  function responseTip(r) {
    return (RESP_NAME[r.action] || r.action) + " (" + r.action + ") from " + (r.agent || "-") + "\nevent " + clockAt(r.t_event) +
      " → done after " + r.latency_s.toFixed(2) + " s (simulated)" + (r.genai_wait_s ? "\nwaited for the LLM " + r.genai_wait_s.toFixed(2) + " s" : "");
  }

  function drawResponseMarks(g, short) {
    var resp = (run().responses || []).slice().sort(function (a, b) { return a.t_done - b.t_done; });
    var ritems = resp.map(function (r) {
      var x = tx(r.t_done), text = (RESP_NAME[r.action] || r.action) + " " + r.latency_s.toFixed(1) + " s";
      return { x0: x - 5, x1: x + 8 + textWidth(text, 11.5), text: text };
    });
    var skip = ritems.map(function (it, i) { return !short && resp[i].action === "A2"; });
    var rrows = greedyRows(ritems.map(function (it, i) { return skip[i] ? { x0: -1e9, x1: -1e9 } : it; }), 3, 4);
    skip.forEach(function (sk, i) { if (sk) rrows[i] = -1; });
    resp.forEach(function (r, i) {
      var row = rrows[i];
      var y = row < 0 ? TL.respY + 47 : TL.respY + 9 + row * 14, x = tx(r.t_done), xe = tx(r.t_event), c = ACTION_COLOR[r.action] || "#fff";
      var k = row < 0 ? 3 : 5, rg = sv("g", null, g);
      FUT.push({ el: rg, t: r.t_done });
      if (row >= 0) sv("line", { x1: xe, y1: y, x2: x, y2: y, stroke: c, "stroke-width": 2, "stroke-opacity": 0.7 }, rg);
      var m = sv("path", { d: "M" + x + " " + (y - k) + " L" + (x + k) + " " + y + " L" + x + " " + (y + k) + " L" + (x - k) + " " + y + " Z", fill: c, stroke: "#0B1E24" }, rg);
      hoverTip(m, responseTip(r));
      if (rrows[i] >= 0) sv("text", { x: x + 8, y: y + 4, "class": "resp-l" }, rg, ritems[i].text);
    });
  }

  function drawTimelineStatic() {
    var svg = $("timeline");
    clear(svg);
    svg.setAttribute("viewBox", "0 0 " + TL.w + " " + TL.h);
    var dur = S.scn.duration_s, short = dur <= 600;
    var g = sv("g", null, svg);
    drawTimelineLanes(g);
    drawTimelineAxis(g, dur);
    drawTruthLane(g, short);
    drawDecisionChips(g, short);
    drawResponseMarks(g, short);
    sv("rect", { "class": "future", x: 0, y: TL.truthY, width: 0, height: TL.axisY - TL.truthY }, svg);
    var ph = sv("g", { "class": "ph" }, svg);
    sv("line", { x1: 0, y1: 0, x2: 0, y2: TL.axisY, "class": "playhead" }, ph);
    sv("path", { d: "M-6 0 L6 0 L0 8 Z", "class": "ph-cap" }, ph);
  }
  function drawPlayhead(t) {
    var svg = $("timeline"), x = tx(t);
    var ph = svg.querySelector(".ph"), fut = svg.querySelector(".future");
    if (ph) ph.setAttribute("transform", "translate(" + x.toFixed(2) + ",0)");
    if (fut) { fut.setAttribute("x", x.toFixed(2)); fut.setAttribute("width", Math.max(0, TL.w - TL.right - x).toFixed(2)); }
    var hide = S.future === "hide";
    FUT.forEach(function (f) { f.el.style.display = hide && f.t > t + 1e-6 ? "none" : ""; });
  }

  /* ---------------------------------------------------------------- right panel */
  var BANNER = {
    none: "ALL CALM", pre_alert: "PRE-ALERT", escalated: "ESCALATED",
    cleared_by_elder: "CLEARED BY THE ELDER", cleared_by_app: "CLEARED IN THE APP"
  };
  function okSaid(text) { return /i'?m ok|i am ok|我沒事/i.test(text); }

  function drawBanner(t) {
    var info = alertInfo(t), b = $("banner"), meter = $("banner-meter");
    b.className = "card banner " + info.state + (info.state === "escalated" && !S.record ? " pulse" : "");
    $("banner-state").textContent = BANNER[info.state] || info.state;
    var sub = "No alert · AhJia is watching";
    meter.className = "banner-meter";
    if (info.setBy && info.state !== "none") {
      var since = " · since " + info.setBy.clock;
      if (info.state === "pre_alert") sub = "Family got a text · " + info.setBy.reason + since;
      else if (info.state === "escalated") sub = "Calling family · " + info.setBy.reason +
        (info.setBy.latency_s !== null ? " · " + info.setBy.latency_s.toFixed(1) + " s (sim)" : "");
      else sub = info.setBy.reason + since;
    }
    if (info.state === "pre_alert" && info.start !== null) {
      var speech = S.scn.world.speech.filter(function (s) { return s.source === "elder" && s.t > info.start + 0.05 && s.t <= t + EPS; });
      if (!speech.length) {
        var left = Math.max(0, info.start + R30 - t);
        meter.className = "banner-meter on";
        $("meter-fill").style.width = (100 * clamp((t - info.start) / R30, 0, 1)).toFixed(1) + "%";
        $("meter-text").textContent = "call in " + Math.ceil(left - 1e-6) + " s unless she answers";
      } else {
        var ok = speech.filter(function (s) { return okSaid(s.text); })[0];
        var pose = elderPoseTimeline(), upStart = null;
        pose.forEach(function (p) {
          if (p[0] > t + EPS) return;
          var up = p[1] === "stand";  // posture from the scenario data; only standing counts
          if (up && upStart === null) upStart = p[0];
          if (!up) upStart = null;
        });
        meter.className = "banner-meter on";
        if (ok && upStart !== null) {
          var from = Math.max(upStart, ok.t), done = clamp(t - from, 0, STAND_S);
          $("meter-fill").style.width = (100 * done / STAND_S).toFixed(1) + "%";
          $("meter-text").textContent = "standing " + done.toFixed(1) + " / " + fmtSec(STAND_S) + " s to clear";
        } else {
          $("meter-fill").style.width = "0%";
          $("meter-text").textContent = "she answered: no call · stays until she stands " + fmtSec(STAND_S) + " s or family confirms";
        }
      }
    }
    $("banner-sub").textContent = sub;
  }

  function hudLines(t) {
    var snap = snapAt(t);
    if (snap && snap.hud) return snap.hud;
    var ds = decisionsUntil(t), d = ds[ds.length - 1];
    if (!d) return ["Agent    : -", "Decision : -", "Reason   : -", "Latency  : -", "Cameras  : -", "GenAI    : -", "Alert    : none"];
    return ["Agent    : " + (d.agent || "-"), "Decision : " + d.actions.join("+"), "Reason   : " + d.reason,
      "Latency  : " + (d.latency_s === null ? "-" : d.latency_s.toFixed(1) + " s (sim)"), "Cameras  : -",
      "GenAI    : " + d.genai, "Alert    : " + d.alert];
  }
  function drawHud(t) {
    var pre = $("hud"), info = alertInfo(t);
    clear(pre);
    hudLines(t).forEach(function (line, i) {
      var k = line.indexOf(":"), key = line.slice(0, k + 1), val = line.slice(k + 1);
      var row = el("span", null, pre);
      el("span", { "class": "k" }, row, key);
      var cls = i === 6 ? "v-alert-" + (info.state.indexOf("cleared") === 0 ? "cleared" : info.state) : "v";
      var m = val.match(/^(.*?)(\(sim\)|sim\))(.*)$/);
      if (m) { el("span", { "class": cls }, row, m[1]); el("span", { "class": "sim" }, row, m[2]); el("span", { "class": cls }, row, m[3]); }
      else el("span", { "class": cls }, row, val);
      pre.appendChild(document.createTextNode("\n"));
    });
  }
  function drawGenai(t) {
    var snap = snapAt(t), ds = decisionsUntil(t), night = isNight(t);
    var st = (snap && snap.genai) || (ds.length ? ds[ds.length - 1].genai : "off");
    var hud = hudLines(t)[5].replace(/^GenAI\s*:\s*/, "");
    var map = { off: ["off", "OFF"], npu_window: ["window", "WINDOW"], loading: ["loading", "LOADING"], host_cpu: ["host", "HOST CPU"] };
    var m = map[st] || ["off", st];
    $("genai").className = "genai " + m[0];
    $("genai-v").textContent = m[1];
    var sub = st === "off" ? (night ? "night: vision only, talk uses template voice" : "vision only, no LLM on the NPU") :
      st === "npu_window" ? hud + " on the NPU" : st === "loading" ? hud : "LLM on the host CPU";
    $("genai-s").textContent = sub;
    var dn = $("daynight");
    clear(dn);
    var ic = sv("svg", { viewBox: "0 0 24 24" }, dn);
    if (night) sv("path", { d: "M15 3a9 9 0 1 0 6 15A8 8 0 0 1 15 3z", fill: "#C9D7F0" }, ic);
    else { sv("circle", { cx: 12, cy: 12, r: 5, fill: "#F2A541" }, ic); sv("path", { d: "M12 1v4M12 19v4M1 12h4M19 12h4M4 4l3 3M17 17l3 3M4 20l3-3M17 7l3-3", stroke: "#F2A541", "stroke-width": 2 }, ic); }
    el("span", null, dn, (night ? "Night " : "Day ") + clockAt(t).slice(0, 5));
  }

  function drawWhatif() {
    var box = $("whatif");
    clear(box);
    DATA.branches.forEach(function (b, i) {
      var btn = el("button", { "class": "wi" + (S.scn.name === b.scenario ? " on" : ""), type: "button" }, box);
      el("span", { "class": "wi-ico" }, btn, String(i + 1));
      var txt = el("span", { "class": "wi-txt" }, btn);
      el("span", null, txt, b.label);
      el("small", null, txt, b.rule || "");
      el("span", { "class": "wi-file" }, btn, "scripted/" + b.scenario + ".yaml");
      btn.addEventListener("click", function () { load(b.scenario, S.strategy); play(); });
    });
    var note = $("scn-note");
    clear(note);
    el("b", null, note, (S.scn.group === "scripted" ? "scripted/" : "scenarios/") + S.scn.name + ".yaml");
    note.appendChild(document.createTextNode(" — " + S.scn.title + " (" + S.scn.source + ")"));
  }

  /* ---------------------------------------------------------------- compare */
  // S2's "delayed by the LLM = 0" follows from the safety-first QoS assumption, not from the simulation.
  var ASSUMED_MARK = " *";
  var ASSUMED_NOTE = "* S2 follows from the safety-first QoS assumption (qos_wait_s = 0); Stage II measures it.";
  function delayAssumed(r) {
    return r.delayed_safety_basis ? String(r.delayed_safety_basis).indexOf("assumption") === 0 : r.strategy === "S2";
  }
  function footnote(box, show) {
    var old = box.querySelector(".fn");
    if (old) old.parentNode.removeChild(old);
    if (show) el("p", { "class": "note fn" }, box, ASSUMED_NOTE);
  }
  function kpiRow(scn, s) {
    return DATA.kpis.rows.filter(function (r) { return r.scenario === scn && r.strategy === s; })[0] || null;
  }
  function cellText(def, r, base) {
    if (!r) return "–";
    return def.f(r, base);
  }
  var KPI_DEFS = [
    { label: "Average power", unit: "W · % of S0", tag: "sim", f: function (r, b) {
      if (r.avg_power_w === null) return "–";
      var pct = b && b.avg_power_w ? Math.round(100 * r.avg_power_w / b.avg_power_w) + " %" : "";
      return [Number(r.avg_power_w).toFixed(2) + " W", pct];
    } },
    { label: "Voice reply p50 / p95 (s)", unit: "", tag: "sim", f: function (r) { return pair(r.voice_p50, r.voice_p95); } },
    { label: "Family text p50 / p95 (s)", unit: "", tag: "sim", f: function (r) { return pair(r.text_p50, r.text_p95); } },
    { label: "Call (A5) at once / after " + fmtSec(R30) + " s (s)", unit: "", tag: "sim", f: function (r) { return pair(r.a5_immediate_s, r.a5_no_answer_s); } },
    { label: "Talk reply p50 (s)", unit: "", tag: "sim", f: function (r) { return num(r.talk_p50, 2) || "–"; } },
    { label: "LLM wakes per hour", unit: "", tag: "sim", f: function (r) { return num(r.llm_wakes_per_h, 1) || "–"; } },
    { label: "Safety events delayed by the LLM", unit: "", f: function (r) { return String(r.delayed_safety) + (delayAssumed(r) ? ASSUMED_MARK : ""); } },
    { label: "High-risk misses", unit: "missed / labelled", f: function (r) { return r.misses + " / " + r.high_risk_labels; } },
    { label: "False alarms", unit: "", day: "per day", f: function (r) { return String(r.false_alarms); } },
    { label: "Cameras per UGen300", unit: "", tag: "est", f: function (r) {
      if (r.cameras_supported === null || r.cameras_supported === undefined) return "–";
      var extra = r.cameras_npu !== undefined ? "NPU " + r.cameras_npu + " · host " + r.cameras_host : "";
      return [String(r.cameras_supported), extra];
    } }
  ];
  function pair(a, b) {
    if ((a === null || a === undefined) && (b === null || b === undefined)) return "–";
    return (num(a, 2) || "–") + " / " + (num(b, 2) || "–");
  }

  function drawKpiTable() {
    var table = $("kpi-table");
    clear(table);
    var groups = [];
    if (S.scn.name !== "day24h" && kpiRow(S.scn.name, "S0")) groups.push({ name: S.scn.name, title: S.scn.friendly, day: false });
    if (kpiRow("day24h", "S0")) groups.push({ name: "day24h", title: "24-hour day (synthetic)", day: true });
    var thead = el("thead", null, table), g1 = el("tr", { "class": "grp" }, thead), g2 = el("tr", null, thead);
    el("th", null, g1, ""); el("th", null, g2, "KPI");
    groups.forEach(function (gr) {
      el("th", { colspan: "3", "class": "sep" }, g1, gr.title);
      ["S0", "S1", "S2"].forEach(function (s, i) { el("th", { "class": i === 0 ? "sep" : "" }, g2, s); });
    });
    var tb = el("tbody", null, table);
    KPI_DEFS.forEach(function (def) {
      var tr = el("tr", null, tb), td = el("td", null, tr);
      el("span", null, td, def.label);
      if (def.tag) el("span", { "class": "tag" + (def.tag === "est" ? " est" : "") }, td, def.tag === "est" ? "estimate" : "simulated");
      groups.forEach(function (gr) {
        var base = kpiRow(gr.name, "S0");
        ["S0", "S1", "S2"].forEach(function (s, i) {
          var c = el("td", { "class": (i === 0 ? "sep " : "") + (s === "S2" ? "s2" : "") }, tr);
          var v = cellText(def, kpiRow(gr.name, s), base);
          if (Array.isArray(v)) { c.appendChild(document.createTextNode(v[0])); if (v[1]) el("span", { "class": "u" }, c, " " + v[1]); }
          else c.textContent = v;
        });
      });
      if (def.unit || (def.day)) {
        var u = def.unit + (def.day && groups.length ? (groups.length > 1 ? "count in the run; " : "") + "per day on the 24 h day" : "");
        el("div", { style: "font-size:12px;color:#7F9693" }, td, u);
      }
    });
    footnote($("kpi-card"), groups.some(function (gr) { var r = kpiRow(gr.name, "S2"); return r && delayAssumed(r); }));
  }

  function barChart(box, opt) {
    clear(box);
    var W = opt.w, H = opt.h, svg = sv("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, box);
    sv("text", { x: 4, y: 20, "class": "chart-title" }, svg, opt.title);
    sv("text", { x: 4, y: 40, "class": "chart-sub" }, svg, opt.sub);
    if (opt.stamp) {
      var st = sv("g", { "class": "stamp", transform: "translate(" + (W - 132) + ",8) rotate(-4)" }, svg);
      sv("rect", { x: 0, y: 0, width: 124, height: 28, rx: 4 }, st);
      sv("text", { x: 62, y: 19, "text-anchor": "middle" }, st, opt.stamp);
    }
    var top = 64, bottom = H - 34, max = opt.max || Math.max.apply(null, opt.values.map(function (v) { return v.v || 0; }).concat([1e-9]));
    var n = opt.values.length, slot = (W - 20) / n, bw = Math.min(110, slot * 0.56);
    sv("line", { x1: 4, y1: bottom, x2: W - 4, y2: bottom, stroke: "#2C4C56" }, svg);
    opt.values.forEach(function (v, i) {
      var cx = 10 + slot * (i + 0.5), hgt = v.v === null ? 0 : (bottom - top - 22) * (v.v / max);
      sv("rect", { x: cx - bw / 2, y: bottom - hgt, width: bw, height: Math.max(hgt, v.v === null ? 0 : 1.5), rx: 5, fill: SERIES[v.s] }, svg);
      sv("text", { x: cx, y: bottom - hgt - 8, "text-anchor": "middle", "class": "bar-v" }, svg, v.text);
      sv("text", { x: cx, y: bottom + 21, "text-anchor": "middle", "class": "bar-n" }, svg, v.s + " " + (v.name || ""));
    });
  }

  function scriptedDelay(v) {
    return { text: String(v.delayed_safety) + (v.delayed_safety_assumed ? ASSUMED_MARK : ""), assumed: !!v.delayed_safety_assumed };
  }

  function drawCompare() {
    drawKpiTable();
    var names = { S0: "fixed", S1: "rules", S2: "Care Router" };
    var day = ["S0", "S1", "S2"].map(function (s) { return kpiRow("day24h", s); });
    if (day[0]) {
      barChart($("chart-power"), { w: 798, h: 306, title: "Average power over the 24-hour day", sub: "Watts on the Hailo-10H, % of S0 · from the replay, not measured",
        stamp: "SIMULATED", values: day.map(function (r, i) {
          var s = ["S0", "S1", "S2"][i];
          return { s: s, name: names[s], v: r ? r.avg_power_w : null,
            text: r ? Number(r.avg_power_w).toFixed(2) + " W · " + Math.round(100 * r.avg_power_w / day[0].avg_power_w) + " %" : "–" };
        }) });
      barChart($("chart-wakes"), { w: 380, h: 276, title: "LLM wakes per hour", sub: "24-hour day · simulated", values: day.map(function (r, i) {
        var s = ["S0", "S1", "S2"][i]; return { s: s, v: r ? r.llm_wakes_per_h : null, text: r ? Number(r.llm_wakes_per_h).toFixed(1) : "–" };
      }) });
      barChart($("chart-cams"), { w: 380, h: 276, title: "Cameras per UGen300", sub: "NPU and host bound · estimate", stamp: "ESTIMATE", values: day.map(function (r, i) {
        var s = ["S0", "S1", "S2"][i]; return { s: s, v: r ? r.cameras_supported : null, text: r ? String(r.cameras_supported) : "–" };
      }) });
    }
    // scripted set
    var sc = DATA.kpis.scripted || {}, box = $("scripted");
    clear(box);
    var any = sc.S0 || sc.S1 || sc.S2;
    $("scripted-sub").textContent = any ? (any.cases + " cases, the same for every strategy") : "";
    var grid = el("div", { "class": "scripted-grid" }, box);
    var t = el("table", { "class": "kpi" }, grid), hd = el("tr", null, el("thead", null, t));
    ["", "S0", "S1", "S2"].forEach(function (h) { el("th", null, hd, h); });
    var tb = el("tbody", null, t);
    [["High-risk misses", function (v) { return v.misses + " / " + v.high_risk_labels; }],
      ["Safety events delayed by the LLM", function (v) { return scriptedDelay(v).text; }],
      ["False alarms", function (v) { return String(v.false_alarms); }]].forEach(function (row) {
      var tr = el("tr", null, tb);
      el("td", null, tr, row[0]);
      ["S0", "S1", "S2"].forEach(function (s) { el("td", { "class": s === "S2" ? "s2" : "" }, tr, sc[s] ? row[1](sc[s]) : "–"); });
    });
    var chart = el("div", null, grid);
    var maxD = Math.max(1, sc.S0 ? sc.S0.delayed_safety : 0, sc.S1 ? sc.S1.delayed_safety : 0, sc.S2 ? sc.S2.delayed_safety : 0);
    barChart(chart, { w: 420, h: 196, title: "Delayed by the LLM", sub: "safety events, scripted set (target 0)", max: maxD,
      values: ["S0", "S1", "S2"].map(function (s) { return { s: s, v: sc[s] ? sc[s].delayed_safety : null, text: sc[s] ? scriptedDelay(sc[s]).text : "–" }; }) });
    footnote($("scripted-card"), ["S0", "S1", "S2"].some(function (s) { return sc[s] && scriptedDelay(sc[s]).assumed; }));
    // spec counts
    var k = DATA.kpis.spec_kinds || {}, card = $("spec-card");
    clear(card);
    el("div", { "class": "disc" }, card, "Simulated, not measured. Stage II measures these on a real UGen300.");
    var kinds = el("div", { "class": "kinds" }, card);
    [["official", "published by Hailo / ASUS"], ["estimate", "derived from cited numbers"], ["assumption", "our placeholder for Stage II"]].forEach(function (p) {
      var d = el("div", { "class": "kind" }, kinds);
      el("b", null, d, String(k[p[0]] || 0));
      d.appendChild(document.createTextNode(p[0] + " · " + p[1]));
    });
    el("div", { "class": "src" }, card, "Simulator inputs: specs/hailo10h.yaml · KPIs: " + (DATA.kpis.source || "none") + " from python -m ahjia replay --all");
  }

  /* ---------------------------------------------------------------- how it works */
  function drawHow() {
    var lad = $("ladder");
    clear(lad);
    ["A0", "A1", "A2", "A3", "A4", "A5"].forEach(function (a) {
      var r = el("div", { "class": "ladder-row" }, lad);
      el("b", { "class": a.toLowerCase(), style: "background:" + ACTION_COLOR[a] }, r, a);
      el("span", null, r, META.actions[a]);
    });
    var f = META.rules.fps, box = $("fps-ladder");
    clear(box);
    [["Empty room", f.empty, fmtFps(f.empty) + "–" + fmtFps(f.empty_watch), "#7A8B90"],
      ["Person present", f.occupied, fmtFps(f.occupied), "#3FA396"],
      ["High-risk zone or time", f.high_risk, fmtFps(f.high_risk), "#F2A541"],
      ["A3 look closer (full-res pose)", FULL, fmtFps(FULL), "#5E93CF"]].forEach(function (row) {
      var r = el("div", { "class": "fps-row" }, box);
      el("span", null, r, row[0]);
      var bar = el("div", null, r);
      el("div", { "class": "bar", style: "width:" + (100 * row[1] / FULL).toFixed(1) + "%;background:" + row[3] }, bar);
      el("span", { "class": "v" }, r, row[2] + " fps");
    });
    $("rule-30").textContent = String(R30);
    $("rule-5").textContent = fmtSec(STAND_S);
  }

  /* ---------------------------------------------------------------- tooltip */
  var stage = $("stage"), tip = $("tip"), scale = 1;
  function showTip(ev, text) { if (S.record) return; tip.textContent = text; tip.hidden = false; moveTip(ev); }
  function moveTip(ev) {
    var r = stage.getBoundingClientRect();
    var x = (ev.clientX - r.left) / scale + 14, y = (ev.clientY - r.top) / scale + 14;
    tip.style.left = Math.min(x, 1920 - 440) + "px"; tip.style.top = Math.min(y, 1080 - 120) + "px";
  }
  function hideTip() { tip.hidden = true; }

  /* ---------------------------------------------------------------- controls */
  var ICON_PLAY = "M5 3 L19 12 L5 21 Z", ICON_PAUSE = "M5 3 H10 V21 H5 Z M14 3 H19 V21 H14 Z";
  function setIcon(btn, d) { clear(btn); var s = sv("svg", { viewBox: "0 0 24 24" }, btn); sv("path", { d: d }, s); }

  function buildControls() {
    var sel = $("scenario");
    clear(sel);
    var g1 = el("optgroup", { label: "Scenarios" }, sel), g2 = el("optgroup", { label: "What if… (scripted)" }, sel);
    var g3 = el("optgroup", { label: "More scripted cases" }, sel);
    var branchOf = {};
    DATA.branches.forEach(function (b) { branchOf[b.scenario] = b; });
    DATA.scenarios.filter(function (s) { return s.main; }).forEach(function (s) { el("option", { value: s.name }, g1, s.friendly); });
    DATA.branches.forEach(function (b) { el("option", { value: b.scenario }, g2, "What if " + b.label); });
    DATA.scenarios.filter(function (s) { return !s.main && !branchOf[s.name]; }).forEach(function (s) { el("option", { value: s.name }, g3, s.title.replace(/^Scripted( probe)?: /, "")); });
    sel.addEventListener("change", function () { load(sel.value, S.strategy); });
    var seg = $("strategy");
    clear(seg);
    DATA.strategies.forEach(function (s) {
      var b = el("button", { type: "button", "data-s": s.name }, seg, s.label);
      if (!S.record) b.title = s.tip;
      b.addEventListener("click", function () { load(S.scn.name, s.name, true); });
    });
    var sp = $("speed");
    clear(sp);
    [["1", "1×"], ["2", "2×"], ["4", "4×"], ["fast", "Fast"]].forEach(function (p) {
      var b = el("button", { type: "button", "data-v": p[0] }, sp, p[1]);
      if (!S.record && p[0] === "fast") b.title = "600× on the 24-hour day, 8× on short scenarios";
      b.addEventListener("click", function () { S.speed = p[0]; syncControls(); });
    });
    $("play").addEventListener("click", function () { if (S.playing) pause(); else play(); });
    $("restart").addEventListener("click", function () { pause(); render(0); });
    setIcon($("restart"), "M12 4 A8 8 0 1 0 20 12 H17 A5 5 0 1 1 12 7 V10 L17 5.5 L12 1 Z");
    $("scrubber").addEventListener("input", function (e) { pause(); render(Number(e.target.value)); });
    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (b) {
      b.addEventListener("click", function () { setTab(b.getAttribute("data-tab")); });
    });
    $("timeline").addEventListener("click", function (ev) {
      var svg = $("timeline"), pt = svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      var p = pt.matrixTransform(svg.getScreenCTM().inverse());
      if (p.x < TL.left) return;
      pause();
      render(clamp((p.x - TL.left) / (TL.w - TL.left - TL.right) * S.scn.duration_s, 0, S.scn.duration_s));
    });
    document.addEventListener("keydown", function (e) {
      if (e.target && e.target.tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); if (S.playing) pause(); else play(); }
      if (e.code === "ArrowRight") { pause(); render(clamp(S.t + 1, 0, S.scn.duration_s)); }
      if (e.code === "ArrowLeft") { pause(); render(clamp(S.t - 1, 0, S.scn.duration_s)); }
    });
  }

  function syncControls() {
    $("scenario").value = S.scn.name;
    Array.prototype.forEach.call($("strategy").children, function (b) { b.className = b.getAttribute("data-s") === S.strategy ? "on" : ""; });
    Array.prototype.forEach.call($("speed").children, function (b) { b.className = b.getAttribute("data-v") === S.speed ? "on" : ""; });
    setIcon($("play"), S.playing ? ICON_PAUSE : ICON_PLAY);
    $("play").className = "icon-btn" + (S.playing ? " playing" : "");
    var sc = $("scrubber");
    sc.max = String(S.scn.duration_s);
    sc.step = S.scn.duration_s > 3600 ? "1" : "0.05";
    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (b) { b.className = "tab" + (b.getAttribute("data-tab") === S.tab ? " on" : ""); });
    ["live", "compare", "how"].forEach(function (v) { $("view-" + v).className = "view" + (v === S.tab ? " on" : ""); });
    $("foot-src").textContent = "Data: python -m ahjia replay --all · " + (S.scn.group === "scripted" ? "scripted/" : "scenarios/") +
      S.scn.name + ".yaml · " + S.strategy + " · " + META.github;
  }

  function setTab(tab) { S.tab = tab; syncControls(); if (tab === "compare") drawCompare(); }

  function load(name, strategy, keepTime) {
    var scn = BY_NAME[name] || S.scn, t = keepTime ? S.t : 0;
    var changed = scn !== S.scn;
    S.scn = scn; S.strategy = strategy || S.strategy;
    if (changed) S.speed = scn.duration_s > 3600 ? "fast" : "1";
    drawTimelineStatic();
    drawWhatif();
    if (S.tab === "compare") drawCompare();
    syncControls();
    render(clamp(t, 0, scn.duration_s));
  }

  function render(t) {
    S.t = t;
    $("clock").textContent = clockAt(t);
    $("tsec").textContent = "t = " + t.toFixed(1) + " s" + (S.scn.synthetic ? " · synthetic day" : "");
    $("scrubber").value = String(t);
    drawPlanDyn(t);
    drawPlayhead(t);
    drawBanner(t);
    drawHud(t);
    drawGenai(t);
  }

  function frame(ts) {
    if (!S.playing) return;
    if (S.last !== null) {
      var t = S.t + (ts - S.last) / 1000 * speedFactor();
      if (t >= S.scn.duration_s) { render(S.scn.duration_s); pause(); return; }
      render(t);
    }
    S.last = ts;
    requestAnimationFrame(frame);
  }
  function play() {
    if (S.t >= S.scn.duration_s - EPS) render(0);
    S.playing = true; S.last = null; syncControls();
    requestAnimationFrame(frame);
  }
  function pause() { S.playing = false; S.last = null; syncControls(); }

  function fit() {
    scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    var ox = (window.innerWidth - 1920 * scale) / 2, oy = (window.innerHeight - 1080 * scale) / 2;
    stage.style.transform = "translate(" + ox + "px," + oy + "px) scale(" + scale + ")";
  }

  /* ---------------------------------------------------------------- legends + boot */
  function legends() {
    var pl = $("plan-legend");
    clear(pl);
    ["empty", "occ", "high", "look"].forEach(function (k) {
      var s = el("span", null, pl); el("i", { style: "background:" + CAM_STATE[k].color }, s); s.appendChild(document.createTextNode("camera " + CAM_STATE[k].label));
    });
    ["elder", "family", "caregiver", "stranger"].forEach(function (k) {
      var s = el("span", null, pl); el("i", { style: "background:" + WHO_COLOR[k] + ";border-radius:50%" }, s); s.appendChild(document.createTextNode(WHO_NAME[k]));
    });
    var tl = $("tl-legend");
    clear(tl);
    el("span", null, tl, "hover a chip for reason and latency · click to jump");
  }

  if (S.record) document.body.classList.add("record"); else document.body.classList.add("anim");
  drawPlanStatic($("plan"));
  legends();
  drawHow();
  buildControls();
  window.addEventListener("resize", fit);
  fit();
  var t0 = Number(Q.get("t") || 0);
  S.speed = S.scn.duration_s > 3600 ? "fast" : "1";
  load(S.scn.name, S.strategy);
  if (["1", "2", "4", "fast"].indexOf(Q.get("speed")) >= 0) { S.speed = Q.get("speed"); syncControls(); }
  setTab(S.tab);
  render(clamp(isFinite(t0) ? t0 : 0, 0, S.scn.duration_s));
  if (Q.get("autoplay") === "1") play();

  /* ---------------------------------------------------------------- recording API */
  window.ahjiaSeek = function (t) {
    pause();
    render(clamp(Number(t) || 0, 0, S.scn.duration_s));
    var info = alertInfo(S.t);
    return { t: S.t, clock: clockAt(S.t), alert: info.state, scenario: S.scn.name, strategy: S.strategy };
  };
  window.ahjiaLoad = function (scenario, strategy, tab) {
    pause();
    load(scenario, strategy);
    if (tab) setTab(tab);
    return { scenario: S.scn.name, strategy: S.strategy, duration_s: S.scn.duration_s };
  };
})();

/* Youreye Vision Dashboard — bağımlılıksız (offline) grafikler ve tablolar */
(() => {
  const D = window.HACKATHON_DATA || {};
  const $ = (s) => document.querySelector(s);
  const root = "../"; // dashboard/ klasöründen proje köküne
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const COL = { cyan: css("--cyan"), violet: css("--violet"), pink: css("--pink"), amber: css("--amber"),
                green: css("--green"), red: css("--red"), muted: css("--muted") };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (v) => `<span class="bar"><b style="width:${Math.round((v || 0) * 100)}%"></b></span><span class="mono">${(v || 0).toFixed(2)}</span>`;

  /* ---------- sekmeler ---------- */
  function show(id) {
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === id));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.target === id));
    requestAnimationFrame(drawAll);
  }
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.target)));
  document.querySelectorAll("[data-goto]").forEach((c) => c.addEventListener("click", () => show(c.dataset.goto)));

  /* ---------- lightbox ---------- */
  const lb = $("#lightbox");
  document.addEventListener("click", (e) => {
    if (e.target.matches("td img")) { $("#lightbox-img").src = e.target.src; lb.classList.add("open"); }
    else if (e.target.closest("#lightbox")) lb.classList.remove("open");
  });

  /* ---------- sayaç animasyonu ---------- */
  function countUp(el) {
    const target = parseFloat(el.dataset.v); if (isNaN(target)) return;
    const dec = (el.dataset.v.split(".")[1] || "").length; const t0 = performance.now();
    const step = (t) => { const k = Math.min(1, (t - t0) / 900); const e = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * e).toFixed(dec); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
  const kpi = (v, l, c) => `<div class="kpi" style="--c:${c}"><div class="v" data-v="${v}">${v}</div><div class="l">${l}</div></div>`;
  function fillKpis(sel, items) { const el = $(sel); el.innerHTML = items.map((i) => kpi(...i)).join(""); el.querySelectorAll(".v").forEach(countUp); }

  /* ---------- canvas yardımcıları ---------- */
  function setup(cv) {
    const r = window.devicePixelRatio || 1; const w = cv.clientWidth || cv.width; const h = cv.clientHeight || cv.height;
    cv.width = w * r; cv.height = h * r; const g = cv.getContext("2d"); g.scale(r, r); return [g, w, h];
  }
  function donut(cv, parts) {
    const [g, w, h] = setup(cv); const total = parts.reduce((a, p) => a + p.v, 0) || 1;
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 10; let a = -Math.PI / 2;
    parts.forEach((p) => { const da = (p.v / total) * Math.PI * 2; g.beginPath(); g.arc(cx, cy, R, a, a + da);
      g.arc(cx, cy, R * 0.62, a + da, a, true); g.closePath(); g.fillStyle = p.c; g.fill(); a += da; });
    g.fillStyle = "#e8ecf7"; g.font = "700 30px JetBrains Mono"; g.textAlign = "center"; g.fillText(parts.reduce((s, p) => s + p.v, 0), cx, cy + 6);
    g.fillStyle = COL.muted; g.font = "13px Outfit"; g.fillText("kişi", cx, cy + 26);
  }
  function bars(cv, items) {
    const [g, w, h] = setup(cv); const max = Math.max(1, ...items.map((i) => i.v)); const bw = (w - 20) / items.length;
    items.forEach((it, i) => { const bh = (h - 60) * (it.v / max); const x = 10 + i * bw + bw * 0.18, y = h - 30 - bh;
      const grd = g.createLinearGradient(0, y, 0, h - 30); grd.addColorStop(0, it.c); grd.addColorStop(1, it.c + "33");
      g.fillStyle = grd; g.beginPath(); g.roundRect(x, y, bw * 0.64, bh, 8); g.fill();
      g.fillStyle = "#e8ecf7"; g.font = "700 15px JetBrains Mono"; g.textAlign = "center"; g.fillText(it.v, x + bw * 0.32, y - 8);
      g.fillStyle = COL.muted; g.font = "13px Outfit"; g.fillText(it.l, x + bw * 0.32, h - 10); });
  }
  function timeline(cv, data, series) {
    const [g, w, h] = setup(cv); if (!data || !data.length) return;
    const pad = { l: 40, r: 14, t: 14, b: 26 }; const tmax = data[data.length - 1].t || 1;
    const vmax = Math.max(1, ...data.flatMap((d) => series.map((s) => d[s.k] || 0)));
    const X = (t) => pad.l + (t / tmax) * (w - pad.l - pad.r); const Y = (v) => h - pad.b - (v / vmax) * (h - pad.t - pad.b);
    g.strokeStyle = "rgba(255,255,255,.06)"; g.fillStyle = COL.muted; g.font = "11px Outfit";
    for (let i = 0; i <= 4; i++) { const v = (vmax * i) / 4; g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(w - pad.r, Y(v)); g.stroke(); g.fillText(Math.round(v), 8, Y(v) + 4); }
    for (let i = 0; i <= 6; i++) { const t = (tmax * i) / 6; g.fillText(`${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`, X(t) - 12, h - 8); }
    series.forEach((s) => {
      g.beginPath(); data.forEach((d, i) => (i ? g.lineTo(X(d.t), Y(d[s.k])) : g.moveTo(X(d.t), Y(d[s.k]))));
      g.strokeStyle = s.c; g.lineWidth = 2.4; g.stroke();
      if (s.fill) { g.lineTo(X(tmax), Y(0)); g.lineTo(X(0), Y(0)); g.closePath(); const grd = g.createLinearGradient(0, pad.t, 0, h);
        grd.addColorStop(0, s.c + "55"); grd.addColorStop(1, s.c + "00"); g.fillStyle = grd; g.fill(); }
    });
    let lx = pad.l + 8; series.forEach((s) => { g.fillStyle = s.c; g.fillRect(lx, pad.t, 10, 10); g.fillStyle = "#cbd5e1"; g.font = "12px Outfit"; g.fillText(s.l, lx + 14, pad.t + 9); lx += g.measureText(s.l).width + 34; });
  }

  /* ---------- Görev 1 ---------- */
  const T1 = D.task1, T2 = D.task2;
  const gBadge = (g) => g === "Erkek" ? `<span class="badge b-m">Erkek</span>` : g === "Kadin" ? `<span class="badge b-f">Kadın</span>` : `<span class="badge b-u">Belirsiz</span>`;
  if (!T1) { $("#t1-empty").style.display = "block"; $("#t1-content").style.display = "none"; }
  else {
    const s = T1.summary, gc = s.gender_crossings || {};
    fillKpis("#t1-kpis", [[s.count_in, "Giriş", COL.green], [s.count_out, "Çıkış", COL.red], [s.total_crossings, "Toplam geçiş", COL.cyan],
      [s.unique_people, "Benzersiz kişi (ID)", COL.violet], [gc.Erkek || 0, "Geçen erkek", "#60a5fa"], [gc.Kadin || 0, "Geçen kadın", COL.pink], [s.avg_fps, "Ortalama FPS", COL.amber]]);
    $("#t1-video").src = root + s.video_out; $("#t1-fps").textContent = `${s.avg_fps} FPS · ${s.resolution}`;
    $("#t1-csv").href = root + s.out_dir + "/crossings.csv"; $("#t1-disclaimer").textContent = "⚠ " + s.disclaimer;
    $("#t1-legend").innerHTML = `<span style="--c:#60a5fa">Erkek ${gc.Erkek || 0}</span><span style="--c:${COL.pink}">Kadın ${gc.Kadin || 0}</span><span style="--c:#64748b">Belirsiz ${gc["?"] || 0}</span>`;
    $("#t1-table tbody").innerHTML = T1.crossings.map((e) => `<tr>
      <td><img src="${root + s.out_dir}/${e.crop}" alt="Kişi ${e.track_id}" loading="lazy" onerror="this.style.display='none'"></td>
      <td class="mono">${esc(e.time)}</td><td class="mono">#${e.track_id}</td>
      <td><span class="badge ${e.direction === "GIRIS" ? "b-in" : "b-out"}">${e.direction === "GIRIS" ? "GİRİŞ" : "ÇIKIŞ"}</span></td>
      <td>${gBadge(e.gender)}</td><td>${pct(e.gender_conf)}</td><td>${pct(e.det_conf)}</td></tr>`).join("");
  }

  /* ---------- Görev 2 ---------- */
  const TYPE_C = { otomobil: COL.cyan, motosiklet: COL.amber, otobus: COL.green, kamyon: COL.red };
  const TYPE_L = { otomobil: "Otomobil", motosiklet: "Motosiklet", otobus: "Otobüs", kamyon: "Kamyon" };
  function renderT2(filter = "") {
    const s = T2.summary; const f = filter.trim().toUpperCase();
    const rows = T2.records.filter((r) => !f || `${r.plate} ${r.vehicle_id} ${r.type}`.toUpperCase().includes(f));
    $("#t2-table tbody").innerHTML = rows.map((r) => `<tr>
      <td><img src="${root + s.out_dir}/${r.vehicle_crop}" alt="Araç ${r.vehicle_id}" loading="lazy" onerror="this.style.display='none'"></td>
      <td>${r.plate_crop ? `<img class="plate" src="${root + s.out_dir}/${r.plate_crop}" alt="Plaka ${esc(r.plate)}" loading="lazy">` : "–"}</td>
      <td class="mono">${esc(r.time)}</td><td class="mono">#${r.vehicle_id}</td>
      <td><span class="badge ${r.direction === "GIRIS" ? "b-in" : "b-out"}">${r.direction === "GIRIS" ? "GİRİŞ" : "ÇIKIŞ"}</span></td>
      <td>${TYPE_L[r.type] || esc(r.type)}</td>
      <td>${r.plate ? `<span class="plate-txt">${esc(r.plate_display)}</span>` : '<span class="muted">okunamadı</span>'}</td>
      <td>${pct(r.plate_ocr_conf)}</td><td>${pct(r.plate_det_conf)}</td><td>${pct(r.vehicle_conf)}</td>
      <td>${r.tr_format_valid ? '<span class="badge b-tr">TR ✓</span>' : ""}${r.repeat_pass ? '<span class="badge b-rep">Tekrar geçiş</span>' : ""}</td></tr>`).join("");
  }
  if (!T2) { $("#t2-empty").style.display = "block"; $("#t2-content").style.display = "none"; }
  else {
    const s = T2.summary;
    fillKpis("#t2-kpis", [[s.count_in, "Şehre giren", COL.green], [s.count_out, "Çıkan", COL.red], [s.total_crossings, "Toplam geçiş", COL.cyan],
      [s.plates_read, "Plaka okunan", COL.violet], [s.plates_tr_valid, "TR formatı geçerli", "#67e8f9"], [s.repeat_passes, "Tekrar geçiş", COL.amber], [s.avg_fps, "Ortalama FPS", COL.amber]]);
    $("#t2-video").src = root + s.video_out; $("#t2-fps").textContent = `${s.avg_fps} FPS · ${s.resolution}`;
    $("#t2-csv").href = root + s.out_dir + "/vehicles.csv"; $("#t2-json").href = root + s.out_dir + "/results.json";
    renderT2(); $("#t2-search").addEventListener("input", (e) => renderT2(e.target.value));
  }

  /* ---------- genel bakış ---------- */
  const hs = [];
  if (T1) hs.push([T1.summary.total_crossings, "Kafe geçişi"], [T1.summary.unique_people, "Takip edilen kişi"]);
  if (T2) hs.push([T2.summary.total_crossings, "Araç geçişi"], [T2.summary.plates_read, "Okunan plaka"]);
  $("#hero-stats").innerHTML = hs.map(([v, l]) => `<div class="hero-stat"><div class="v" data-v="${v}">${v}</div><div class="l">${l}</div></div>`).join("") || '<p class="muted">Henüz sonuç yok.</p>';
  $("#hero-stats").querySelectorAll(".v").forEach(countUp);
  const mini = (v, l, c) => `<div class="kpi" style="--c:${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  if (T1) { const s = T1.summary; $("#ov1-fps").textContent = `${s.avg_fps} FPS`;
    $("#ov1-kpis").innerHTML = mini(s.count_in, "Giriş", COL.green) + mini(s.count_out, "Çıkış", COL.red) + mini((s.gender_crossings || {}).Erkek || 0, "Erkek", "#60a5fa") + mini((s.gender_crossings || {}).Kadin || 0, "Kadın", COL.pink); }
  if (T2) { const s = T2.summary; $("#ov2-fps").textContent = `${s.avg_fps} FPS`;
    $("#ov2-kpis").innerHTML = mini(s.count_in, "Şehre giren", COL.green) + mini(s.total_crossings, "Toplam", COL.cyan) + mini(s.plates_read, "Plaka", COL.violet) + mini(s.repeat_passes, "Tekrar", COL.amber); }

  function drawAll() {
    if (T1 && $("#task1").classList.contains("active")) {
      const gc = T1.summary.gender_crossings || {};
      donut($("#t1-donut"), [{ v: gc.Erkek || 0, c: "#60a5fa" }, { v: gc.Kadin || 0, c: COL.pink }, { v: gc["?"] || 0, c: "#475569" }]);
      timeline($("#t1-timeline"), T1.timeline, [{ k: "in", l: "Giriş (kümülatif)", c: COL.green }, { k: "out", l: "Çıkış (kümülatif)", c: COL.red }, { k: "in_frame", l: "Kadrajdaki kişi", c: COL.cyan, fill: true }]);
    }
    if (T2 && $("#task2").classList.contains("active")) {
      const ty = T2.summary.types || {};
      bars($("#t2-bars"), Object.keys(TYPE_L).map((k) => ({ v: ty[k] || 0, l: TYPE_L[k], c: TYPE_C[k] })));
      timeline($("#t2-timeline"), T2.timeline, [{ k: "in", l: "Şehre giren (kümülatif)", c: COL.green }, { k: "out", l: "Çıkan", c: COL.red }, { k: "in_frame", l: "Kadrajdaki araç", c: COL.amber, fill: true }]);
    }
  }
  window.addEventListener("resize", drawAll);
  drawAll();
})();

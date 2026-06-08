use actix_web::{web, App, HttpServer, HttpResponse};
use actix_files as fs;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::LazyLock;

// ---- data model ----

#[derive(Serialize, Clone)]
struct ImageEntry {
    path: String,   // URL path, e.g. "imgs/100870710_p0.png"
    label: String,  // method name, e.g. "original", "dct_heat"
    source: String, // parent dir, e.g. "imgs", "dct_viz"
}

#[derive(Serialize, Clone)]
struct Group {
    prefix: String, // e.g. "100870710_p0"
    images: Vec<ImageEntry>,
}

// ---- prefix parsing ----
// stem: "100870710_p0"              -> (prefix="100870710_p0", method="original")
// stem: "100870710_p0_dct_heat"     -> (prefix="100870710_p0", method="dct_heat")
// stem: "100870710_p0_sr_overlay"   -> (prefix="100870710_p0", method="sr_overlay")

fn parse_stem(stem: &str) -> (String, String) {
    let p = stem.find("_p").unwrap_or(stem.len());
    if p == stem.len() {
        return (stem.to_string(), "original".to_string());
    }
    let after = &stem[p + 2..];
    let dlen = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
    let pre_end = p + 2 + dlen;
    let prefix = stem[..pre_end].to_string();
    let method = if pre_end < stem.len() && stem.as_bytes().get(pre_end) == Some(&b'_') {
        stem[pre_end + 1..].to_string()
    } else {
        "original".to_string()
    };
    (prefix, method)
}

// ---- scan groups ----

fn scan_groups() -> Vec<Group> {
    let mut map: BTreeMap<String, Vec<ImageEntry>> = BTreeMap::new();

    // 1) imgs/
    if let Ok(entries) = std::fs::read_dir("imgs") {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let stem = name.rsplitn(2, '.').nth(1).unwrap_or(&name);
            let (prefix, _) = parse_stem(stem);
            map.entry(prefix).or_default().push(ImageEntry {
                path: format!("imgs/{}", name),
                label: "original".to_string(),
                source: "imgs".to_string(),
            });
        }
    }

    // 2) output/ subdirs
    if let Ok(dirs) = std::fs::read_dir("output") {
        for d in dirs.flatten() {
            let Ok(ft) = d.file_type() else { continue };
            if !ft.is_dir() { continue; }
            let src = d.file_name().to_string_lossy().to_string();
            let sub = format!("output/{src}");
            if let Ok(files) = std::fs::read_dir(&sub) {
                for f in files.flatten() {
                    let name = f.file_name().to_string_lossy().to_string();
                    let stem = name.rsplitn(2, '.').nth(1).unwrap_or(&name);
                    let (prefix, method) = parse_stem(stem);
                    map.entry(prefix).or_default().push(ImageEntry {
                        path: format!("{sub}/{name}"),
                        label: method,
                        source: src.clone(),
                    });
                }
            }
        }
    }

    map.into_iter().map(|(k, v)| Group { prefix: k, images: v }).collect()
}

static GROUPS: LazyLock<Vec<Group>> = LazyLock::new(scan_groups);

// ---- handlers ----

async fn index() -> HttpResponse {
    HttpResponse::Ok().content_type("text/html; charset=utf-8").body(HTML)
}

async fn get_groups() -> HttpResponse {
    HttpResponse::Ok().json(&*GROUPS)
}

// ---- main ----

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Scanning image groups...");
    let n = GROUPS.len();
    println!("Found {n} groups. Serving at http://127.0.0.1:3000");

    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
            .route("/api/groups", web::get().to(get_groups))
            .service(fs::Files::new("/imgs", "imgs").show_files_listing().use_last_modified(true))
            .service(fs::Files::new("/output", "output").show_files_listing().use_last_modified(true))
    })
    .bind("127.0.0.1:3000")?
    .run()
    .await
}

// ---- embedded HTML ----

const HTML: &str = r##"<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>图片对比合成</title>
<style>
:root {
    --bg: #0f0f1a;
    --panel: #151528;
    --card: #1d1d35;
    --card-sel: #252545;
    --text: #ddd;
    --dim: #888;
    --accent: #6c63ff;
    --border: #2a2a4a;
    --radius: 8px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif; height:100vh; display:flex; overflow:hidden; }

/* ---- left panel ---- */
#left {
    width:240px; min-width:240px; background:var(--panel); border-right:1px solid var(--border);
    display:flex; flex-direction:column; overflow:hidden;
}
#left h3 { padding:16px 12px 8px; font-size:13px; color:var(--dim); text-transform:uppercase; letter-spacing:1px; }
#search { margin:0 8px 8px; padding:6px 10px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); font-size:13px; outline:none; }
#search:focus { border-color:var(--accent); }
#group-list { flex:1; overflow-y:auto; padding:0 4px; }
.group-item {
    display:flex; align-items:center; gap:6px; padding:8px 10px; margin:2px 0;
    border-radius:6px; cursor:pointer; font-size:13px; transition:background .15s;
}
.group-item:hover { background:var(--card); }
.group-item.active { background:var(--accent); color:#fff; }
.group-item .badge { font-size:10px; background:var(--border); color:var(--dim); padding:1px 6px; border-radius:10px; margin-left:auto; }

/* ---- right panel ---- */
#right { flex:1; display:flex; flex-direction:column; overflow:hidden; }

/* tool bar */
#toolbar {
    display:flex; align-items:center; gap:12px; padding:10px 16px;
    border-bottom:1px solid var(--border); background:var(--panel); flex-wrap:wrap;
}
#toolbar label { font-size:13px; color:var(--dim); }
#toolbar select, #toolbar input[type=range] { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:4px 8px; font-size:13px; }
#toolbar button {
    padding:6px 14px; border:none; border-radius:6px; background:var(--accent); color:#fff;
    font-size:13px; cursor:pointer; transition:opacity .15s;
}
#toolbar button:hover { opacity:.85; }

/* image grid */
#grid-wrap { flex:1; overflow-y:auto; padding:12px; }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:10px; }
.card {
    background:var(--card); border-radius:var(--radius); overflow:hidden;
    border:2px solid transparent; transition:border-color .15s; cursor:pointer;
}
.card.selected { border-color:var(--accent); background:var(--card-sel); }
.card img { width:100%; height:180px; object-fit:contain; background:#000; display:block; }
.card .info { padding:8px 10px; display:flex; align-items:center; gap:8px; font-size:12px; }
.card .info input[type=checkbox] { accent-color:var(--accent); width:16px; height:16px; cursor:pointer; }
.card .fname { flex:1; word-break:break-all; color:var(--dim); font-size:11px; }
.card .method { background:var(--border); color:var(--text); padding:2px 7px; border-radius:4px; font-size:11px; white-space:nowrap; }

/* composite section */
#composite-section { border-top:1px solid var(--border); background:var(--panel); padding:12px 16px; max-height:320px; overflow-y:auto; }
#composite-section h4 { font-size:14px; margin-bottom:8px; }
#layer-weights { display:flex; flex-direction:column; gap:6px; margin:8px 0; }
.layer-row { display:flex; align-items:center; gap:8px; font-size:12px; }
.layer-row span:first-child { width:90px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--dim); }
.layer-row input[type=range] { flex:1; accent-color:var(--accent); }
.layer-row .wval { width:36px; text-align:right; color:var(--text); }
#composite-canvas { max-width:100%; max-height:200px; border-radius:6px; border:1px solid var(--border); margin-top:8px; }

/* file type indicator */
.source-tag { font-size:10px; padding:1px 5px; border-radius:3px; background:var(--accent); color:#fff; margin-left:4px; }
</style>
</head>
<body>

<div id="left">
    <h3>📁 图片分组</h3>
    <input id="search" type="text" placeholder="搜索前缀..." oninput="filterGroups()">
    <div id="group-list"></div>
</div>

<div id="right">
    <div id="toolbar">
        <label>合成模式:</label>
        <select id="blend-mode" onchange="recomposite()">
            <option value="add">相加 (Add)</option>
            <option value="multiply">相乘 (Multiply)</option>
        </select>
        <label>全局权重:</label>
        <input type="range" id="global-weight" min="0" max="200" value="100" oninput="updateGlobalWeight();recomposite()" style="width:100px;">
        <span id="global-weight-val" style="font-size:12px;color:var(--dim);">1.00</span>
        <label>源阈值:</label>
        <input type="range" id="threshold" min="0" max="255" value="0" oninput="updateThreshold();recomposite()" style="width:80px;">
        <span id="threshold-val" style="font-size:12px;color:var(--dim);">0</span>
        <label>结果阈值:</label>
        <input type="range" id="result-threshold" min="0" max="255" value="0" oninput="updateResultThreshold();recomposite()" style="width:80px;">
        <span id="result-threshold-val" style="font-size:12px;color:var(--dim);">0</span>
        <button onclick="toggleAll(true)">全选</button>
        <button onclick="toggleAll(false)">全不选</button>
        <button onclick="recomposite()">🔃 刷新合成</button>
    </div>
    <div id="grid-wrap"><div id="grid"></div></div>
    <div id="composite-section">
        <h4>🖼 合成结果</h4>
        <div id="layer-weights"></div>
        <canvas id="composite-canvas" style="display:none"></canvas>
        <div id="no-composite" style="color:var(--dim);font-size:13px;">请勾选至少一张图片进行合成</div>
    </div>
</div>

<script>
// ---- state ----
let groups = [];
let activePrefix = null;
let activeImages = [];
let imageCache = {};  // path -> HTMLImageElement

// ---- init ----
fetch('/api/groups').then(r=>r.json()).then(data=>{
    groups = data;
    renderGroupList();
    if (groups.length) selectGroup(groups[0].prefix);
});

function renderGroupList(filter) {
    const list = document.getElementById('group-list');
    list.innerHTML = groups
        .filter(g => !filter || g.prefix.includes(filter))
        .map(g => `<div class="group-item${g.prefix===activePrefix?' active':''}"
            onclick="selectGroup('${g.prefix}')">
            ${g.prefix}<span class="badge">${g.images.length}</span></div>`).join('');
}

function filterGroups() {
    renderGroupList(document.getElementById('search').value);
}

function selectGroup(prefix) {
    activePrefix = prefix;
    document.querySelectorAll('.group-item').forEach(el=>{
        el.classList.toggle('active', el.textContent.trim().startsWith(prefix));
    });
    const g = groups.find(x=>x.prefix===prefix);
    if (!g) return;
    activeImages = g.images;
    renderGrid();
    renderLayerWeights();
    document.getElementById('grid-wrap').scrollTop = 0;
}

// ---- grid ----
function renderGrid() {
    const grid = document.getElementById('grid');
    grid.innerHTML = activeImages.map((img,i)=>`
        <div class="card" id="card-${i}" onclick="onCardClick(${i},event)">
            <img src="${img.path}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22280%22 height=%22180%22><rect fill=%22%23333%22 width=%22280%22 height=%22180%22/><text fill=%22%23888%22 x=%22140%22 y=%2290%22 text-anchor=%22middle%22 font-size=%2214%22>Load Failed</text></svg>'">
            <div class="info">
                <input type="checkbox" id="cb-${i}" onchange="onCheck(${i})" onclick="event.stopPropagation()">
                <span class="method">${img.label}</span>
                <span class="source-tag">${img.source}</span>
            </div>
        </div>`).join('');

    // preload images
    activeImages.forEach((img,i) => {
        if (!imageCache[img.path]) {
            const el = new Image();
            el.crossOrigin = 'anonymous';
            el.src = img.path;
            imageCache[img.path] = el;
        }
    });
}

function onCardClick(i, ev) {
    if (ev.target.tagName === 'INPUT') return;
    const cb = document.getElementById('cb-'+i);
    cb.checked = !cb.checked;
    onCheck(i);
}

function onCheck(i) {
    const cb = document.getElementById('cb-'+i);
    document.getElementById('card-'+i).classList.toggle('selected', cb.checked);
    renderLayerWeights();
    recomposite();
}

function updateGlobalWeight() {
    const v = document.getElementById('global-weight').value/100;
    document.getElementById('global-weight-val').textContent = v.toFixed(2);
}

function updateThreshold() {
    const v = document.getElementById('threshold').value;
    document.getElementById('threshold-val').textContent = v;
}

function updateResultThreshold() {
    const v = document.getElementById('result-threshold').value;
    document.getElementById('result-threshold-val').textContent = v;
}

function toggleAll(on) {
    document.querySelectorAll('#grid input[type=checkbox]').forEach(cb=>{ cb.checked = on; });
    document.querySelectorAll('.card').forEach(c=>c.classList.toggle('selected', on));
    renderLayerWeights();
    recomposite();
}

// ---- layer weights ----
function renderLayerWeights() {
    const container = document.getElementById('layer-weights');
    const checked = [];
    document.querySelectorAll('#grid input[type=checkbox]').forEach((cb,i)=>{
        if (cb.checked) checked.push(i);
    });
    if (!checked.length) { container.innerHTML=''; return; }
    container.innerHTML = checked.map(i=>{
        const img = activeImages[i];
        return `<div class="layer-row">
            <span title="${img.path}">${img.label} <small>${img.source}</small></span>
            <input type="range" min="0" max="200" value="100" oninput="updateWeight(this,${i})">
            <span class="wval">1.00</span>
        </div>`;
    }).join('');
}

function updateWeight(slider, i) {
    slider.nextElementSibling.textContent = (slider.value/100).toFixed(2);
    recomposite();
}

function getCheckedLayers() {
    const layers = [];
    document.querySelectorAll('#grid input[type=checkbox]').forEach((cb,i)=>{
        if (cb.checked) {
            const row = document.querySelectorAll('.layer-row')[layers.length];
            const w = row ? row.querySelector('input[type=range]').value/100 : 1;
            layers.push({index:i, weight:w});
        }
    });
    return layers;
}

// ---- compositing ----
function recomposite() {
    const layers = getCheckedLayers();
    const canvas = document.getElementById('composite-canvas');
    const noComp = document.getElementById('no-composite');

    if (layers.length === 0) {
        canvas.style.display = 'none';
        noComp.style.display = '';
        return;
    }
    noComp.style.display = 'none';
    canvas.style.display = '';

    const globalW = document.getElementById('global-weight').value/100;
    document.getElementById('global-weight-val').textContent = globalW.toFixed(2);

    const mode = document.getElementById('blend-mode').value;

    // Wait for all images to load
    const imgs = layers.map(l => imageCache[activeImages[l.index].path]).filter(Boolean);
    let loaded = 0;
    imgs.forEach(el => {
        if (el.complete && el.naturalWidth) loaded++;
        else el.onload = el.onerror = ()=>{ loaded++; if (loaded===imgs.length) doComposite(); };
    });
    if (loaded === imgs.length) doComposite();

    function doComposite() {
        // Determine canvas size (use first valid image dimensions)
        let w=0, h=0;
        for (const el of imgs) {
            if (el.naturalWidth && el.naturalHeight) { w=el.naturalWidth; h=el.naturalHeight; break; }
        }
        if (!w || !h) return;

        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');

        // Draw each layer into a temp canvas, then composite
        const tmp = document.createElement('canvas');
        tmp.width = w; tmp.height = h;
        const tctx = tmp.getContext('2d');

        // Buffer for accumulation
        const result = ctx.createImageData(w, h);
        const rdata = result.data;

        // For multiply: start with white; for add: start with black
        if (mode === 'multiply') {
            for (let i=0; i<rdata.length; i+=4) {
                rdata[i]=255; rdata[i+1]=255; rdata[i+2]=255; rdata[i+3]=255;
            }
        } else {
            for (let i=0; i<rdata.length; i+=4) {
                rdata[i]=0; rdata[i+1]=0; rdata[i+2]=0; rdata[i+3]=255;
            }
        }

        const thr = document.getElementById('threshold').value;

        for (let li=0; li<layers.length; li++) {
            const el = imgs[li];
            if (!el.naturalWidth) continue;
            const lw = layers[li].weight * globalW;
            tctx.clearRect(0,0,w,h);
            tctx.drawImage(el, 0, 0, w, h);
            const src = tctx.getImageData(0,0,w,h).data;

            if (mode==='add') {
                for (let p=0; p<rdata.length; p+=4) {
                    const bright = 0.299*src[p] + 0.587*src[p+1] + 0.114*src[p+2];
                    if (bright < thr) continue;
                    rdata[p]   = Math.min(255, rdata[p]   + src[p]   * lw);
                    rdata[p+1] = Math.min(255, rdata[p+1] + src[p+1] * lw);
                    rdata[p+2] = Math.min(255, rdata[p+2] + src[p+2] * lw);
                }
            } else { // multiply
                for (let p=0; p<rdata.length; p+=4) {
                    const bright = 0.299*src[p] + 0.587*src[p+1] + 0.114*src[p+2];
                    const f = bright < thr ? 0 : 1; // below threshold: skip this layer's contribution
                    // weighted multiply: result *= (src/255)^(lw*f)
                    const wgt = lw * f;
                    rdata[p]   *= Math.pow(src[p]   / 255, wgt);
                    rdata[p+1] *= Math.pow(src[p+1] / 255, wgt);
                    rdata[p+2] *= Math.pow(src[p+2] / 255, wgt);
                }
            }
        }

        // 结果阈值: 亮度低于阈值的像素置黑
        const rthr = document.getElementById('result-threshold').value;
        if (rthr > 0) {
            for (let p=0; p<rdata.length; p+=4) {
                const bright = 0.299*rdata[p] + 0.587*rdata[p+1] + 0.114*rdata[p+2];
                if (bright < rthr) {
                    rdata[p]=0; rdata[p+1]=0; rdata[p+2]=0;
                }
            }
        }

        ctx.putImageData(result, 0, 0);

        // Scale display
        const maxH = 200;
        if (h > maxH) {
            const scale = maxH/h;
            canvas.style.width = (w*scale)+'px';
            canvas.style.height = maxH+'px';
        }
    }
}
</script>
</body>
</html>"##;

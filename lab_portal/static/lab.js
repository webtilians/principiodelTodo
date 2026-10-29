'use strict';
const $ = (id) => document.getElementById(id);
const accessKey = 'infinito-lab-token';
function savedAccess() {
  try { return localStorage.getItem(accessKey) || sessionStorage.getItem(accessKey) || ''; }
  catch { return ''; }
}
function forgetAccess() {
  try { localStorage.removeItem(accessKey); sessionStorage.removeItem(accessKey); } catch {}
}
const state = {token: savedAccess(), catalog: [], selected: null,
               jobs: [], job: null, poll: null, renderedReport: null};
const icons = {f4:'◇', f5:'◈', f6:'⌁', f7:'⊙', f7b:'⟡', f8:'◎', f9_memory:'▣', f9_doorkey:'⚿', f9_dynamic:'↝'};
const labels = {f4:'MEMORIA', f5:'PREDICCIÓN', f6:'MODELO / RL', f7:'INFORMACIÓN', f7b:'RECALIBRACIÓN', f8:'MUNDO ACTIVO', f9_memory:'EXTERNO: MEMORIA', f9_doorkey:'EXTERNO: PLAN', f9_dynamic:'EXTERNO: DINÁMICA'};
function elem(tag, className, content) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (content !== undefined) el.textContent = content;
  return el;
}
function notify(message) {
  $('toast').textContent = message;
  $('toast').classList.add('show');
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => $('toast').classList.remove('show'), 4200);
}
async function api(path, options = {}) {
  const headers = Object.assign({'X-Lab-Token': state.token}, options.headers || {});
  if (options.body) headers['Content-Type'] = 'application/json';
  let response;
  try { response = await fetch('/api/' + path, {...options, headers, cache:'no-store'}); }
  catch (e) { throw Error('No se puede contactar con el servidor local.'); }
  if (response.status === 401) {
    forgetAccess();
    if (!$('auth-modal').hidden) throw Error('Clave incorrecta');
    $('auth-modal').hidden = false;
    state.token = '';
    forgetAccess();
    throw Error('Sesión caducada o clave incorrecta');
  }
  if (!response.ok) {
    let error;
    try { error = (await response.json()).detail; } catch { error = response.statusText; }
    throw Error(Array.isArray(error) ? error.map(x => x.msg).join('; ') : (error || 'Error ' + response.status));
  }
  return response;
}
async function json(path, options = {}) { return (await api(path, options)).json(); }
async function download(path, suggested) {
  const resp = await api(path);
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = elem('a'); a.href = url; a.download = suggested;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 6000);
}
function setStatus(online) {
  $('server-status').classList.toggle('online', online);
  $('server-status').innerHTML = '';
  const dot = elem('i'); $('server-status').append(dot, document.createTextNode(online ? ' CONECTADO' : ' SIN CONEXIÓN'));
}
function expById(id) { return state.catalog.find(x => x.id === id); }
function selectExperiment(id) {
  state.selected = id;
  const exp = expById(id);
  if (!exp) return;
  document.querySelectorAll('.module-option').forEach(x => x.classList.toggle('selected', x.dataset.experiment === id));
  document.querySelectorAll('.experiment-card').forEach(x => x.classList.toggle('active', x.dataset.experiment === id));
  $('current-exp').textContent = exp.name;
  $('exp-description').textContent = exp.description;
  $('exp-warning').textContent = exp.warning;
  const fields = $('fields'); fields.replaceChildren();
  for (const [name, descriptor] of Object.entries(exp.params)) {
    const row = elem('div','field'); const label=elem('label'); label.htmlFor='param-'+name;
    label.append(elem('span','',descriptor.label), elem('small','',`${descriptor.minimum}–${descriptor.maximum}`));
    const input=elem('input'); input.id='param-'+name; input.type='number';input.required=true;
    input.name=name;input.min=descriptor.minimum;input.max=descriptor.maximum;
    input.step=descriptor.step;input.value=descriptor.default;
    row.append(label,input);fields.append(row);
  }
  setPreset('estandar');
}
function setPreset(preset) {
  document.querySelectorAll('[data-preset]').forEach(x => x.classList.toggle('active', x.dataset.preset === preset));
  const exp=expById(state.selected);
  if (!exp) return;
  for (const [name, value] of Object.entries(exp.presets[preset] || {})) {
    const input=$('param-'+name);
    if (input) input.value=value;
  }
}
function drawCatalog() {
  const cards=$('experiment-cards'), modules=$('modules-nav');
  cards.replaceChildren();modules.replaceChildren();
  for (const exp of state.catalog) {
    const card=elem('button','experiment-card');card.type='button';card.dataset.experiment=exp.id;
    const icon=elem('div','card-icon',icons[exp.id]);
    const copy=elem('div','card-copy');copy.append(elem('strong','',exp.name),elem('span','',exp.description));
    card.append(icon,copy,elem('div','card-check','↗'));
    card.onclick=()=>selectExperiment(exp.id); cards.append(card);
    const item=elem('button','module-option',`${exp.id.toUpperCase()}   ${labels[exp.id]}`);
    item.dataset.experiment=exp.id; item.onclick=()=>{selectExperiment(exp.id);window.location.hash='#experiments';};
    modules.append(item);
  }
  $('stat-modules').textContent=state.catalog.length;
  selectExperiment(state.catalog[0]?.id);
}
function prettyDate(date) {
  if (!date) return 'Sin fecha';
  try {return new Intl.DateTimeFormat('es-ES', {day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}).format(new Date(date));}
  catch {return date;}
}
function renderRuns() {
  $('stat-runs').textContent=state.jobs.length;
  const active=state.jobs.filter(x=>['running','queued'].includes(x.status));
  $('stat-running').textContent=active.length;
  $('launch').disabled=active.length>0;
  $('launch').title=active.length?'Ya hay un experimento activo':'';
  const target=$('runs-list');target.replaceChildren();
  if (!state.jobs.length) {target.append(elem('div','empty','Todavía no hay ejecuciones.'));return;}
  for(const job of state.jobs){
    const row=elem('button','run-row'+(job.id===state.job?' active':''));row.type='button';
    const dot=elem('div','run-status-dot '+job.status);
    const meta=elem('div','run-copy');
    meta.append(elem('strong','',`${job.experiment.toUpperCase()}  ·  ${job.status.toUpperCase()}`),elem('small','',prettyDate(job.created_at)));
    row.append(dot,meta,elem('div','run-shortid',job.id.slice(0,8)));
    row.onclick=async()=>{state.job=job.id;renderRuns();await refreshSelected();};
    target.append(row);
  }
}
async function refreshJobs() {
  const data=await json('jobs');state.jobs=data.jobs;renderRuns();
  if (!state.job && data.jobs.length) state.job=data.jobs[0].id;
  if(state.job) await refreshSelected();
}
async function refreshSelected() {
  const id=state.job;
  if(!id)return;
  const [job, logs] = await Promise.all([json('jobs/'+id),json('jobs/'+id+'/log')]);
  if (id !== state.job) return;
  $('detail-title').textContent=job.experiment.toUpperCase()+' / '+job.id.slice(0,8);
  const pill=$('job-state');pill.textContent=job.status.toUpperCase();pill.className='pill '+job.status;
  $('job-meta').textContent=`Creado ${prettyDate(job.created_at)}  ·  ${Object.entries(job.params).map(([k,v])=>k+'='+v).join('  /  ')}`;
  $('terminal').textContent=logs.log||'$ esperando salida…';
  $('cancel').disabled=!['running','queued'].includes(job.status);
  $('download').disabled=!['completed','failed','timeout','cancelled','interrupted'].includes(job.status);
  $('report').disabled=job.status!=='completed';
  if (job.status==='completed' && state.renderedReport !== id) {
    try {const data=await json('jobs/'+id+'/summary');renderSummary(data);state.renderedReport=id;}
    catch{clearSummary();}
  } else if(job.status!=='completed' && state.renderedReport!==id){clearSummary();}
}
function clearSummary() {
  $('results-empty').hidden=false;$('results-data').hidden=true;state.renderedReport=null;
}
function renderSummary(summary) {
  $('results-empty').hidden=true;$('results-data').hidden=false;
  $('metric-title').textContent=summary.label;
  const scenarios=Object.keys(summary.scenarios);const select=$('scenario');select.replaceChildren();
  for(const scenario of scenarios){const option=elem('option','',scenario.replace(/_/g,' '));option.value=scenario;select.append(option);}
  select.onchange=()=>renderBars(summary,select.value);
  if(scenarios.length) renderBars(summary,scenarios[0]);
}
function renderBars(summary,scenario) {
  const values=Object.entries(summary.scenarios[scenario]||{}).filter(([_,v])=>Number.isFinite(v));
  const area=$('bars');area.replaceChildren();
  const max=Math.max(0.01,...values.map(([_,v])=>Math.abs(v*summary.scale)));
  for(const [name,raw] of values){
    const value=raw*summary.scale;
    const row=elem('div','bar-row');const label=elem('span','bar-label',name.replace(/_/g,' '));
    const track=elem('div','bar-track');const bar=elem('div','bar-fill');
    bar.style.width=Math.max(1,Math.abs(value)/max*100)+'%';
    if(value<0) bar.style.background='linear-gradient(90deg,#ad6969,#dba69c)';
    track.append(bar);row.append(label,track,elem('span','bar-value',value.toFixed(value>100?1:2)));
    area.append(row);
  }
  $('metric-note').textContent=`Escenario: ${scenario.replace(/_/g,' ')}. Valores del JSON de esta ejecución. Las escalas no deben compararse entre protocolos. ${summary.unit==='%'?'Porcentajes sobre 100.':''}`;
}
async function load() {
  try{
    await json('ping');setStatus(true);
    try { localStorage.setItem(accessKey, state.token); sessionStorage.removeItem(accessKey); }
    catch { notify('El navegador no permite recordar el acceso en este dispositivo.'); }
    $('token-input').value='';
    state.catalog=(await json('catalog')).experiments;
    $('auth-modal').hidden=true;drawCatalog();await refreshJobs();
    clearInterval(state.poll);
    state.poll=setInterval(async()=>{
      try {await refreshJobs();setStatus(true);} catch{setStatus(false);}
    },3000);
  }catch(error){setStatus(false);$('auth-modal').hidden=false;$('auth-error').textContent=error.message;}
}
$('auth-form').onsubmit=async event=>{
  event.preventDefault();state.token=$('token-input').value.trim();
  $('auth-error').textContent='';await load();
};
$('logout').onclick=()=>{clearInterval(state.poll);forgetAccess();state.token='';$('token-input').value='';setStatus(false);$('auth-modal').hidden=false;};
document.querySelectorAll('[data-preset]').forEach(btn=>btn.onclick=()=>setPreset(btn.dataset.preset));
$('run-form').onsubmit=async event=>{
  event.preventDefault();const exp=expById(state.selected);if(!exp)return;
  const params={};
  for(const key of Object.keys(exp.params)) params[key]=Number($('param-'+key).value);
  try{
    $('launch').disabled=true;
    const created=await json('jobs',{method:'POST',body:JSON.stringify({experiment:exp.id,params})});
    state.job=created.id;state.renderedReport=null;clearSummary();
    notify('Experimento '+created.id.slice(0,8)+' iniciado');await refreshJobs();
    window.location.hash='#runs';
  }catch(error){notify(error.message);$('launch').disabled=false;}
};
$('cancel').onclick=async()=>{
  if(!state.job || !confirm('¿Cancelar esta ejecución? Los resultados parciales se conservarán.')) return;
  try{await json('jobs/'+state.job+'/cancel',{method:'POST'});await refreshJobs();notify('Se solicitó detener la ejecución.');}
  catch(e){notify(e.message);}
};
$('download').onclick=async()=>{if(!state.job)return;try{await download('jobs/'+state.job+'/download','infinito_lab_'+state.job.slice(0,8)+'.zip');}catch(e){notify(e.message);}};
$('report').onclick=async()=>{if(!state.job)return;try{await download('jobs/'+state.job+'/report','infinito_'+state.job.slice(0,8)+'_report.json');}catch(e){notify(e.message);}};
$('refresh').onclick=()=>refreshJobs().catch(e=>notify(e.message));
if(state.token)load();

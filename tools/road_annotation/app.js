'use strict';
const $ = id => document.getElementById(id);
let plan, entries, index = 0, pending = [], image = null, loaded = false, dirty = false;
const colours = {0:'#ff8d8d',1:'#82e4b2',255:'#ffd797'}, names = {0:'Non-road cutout',1:'Visible road',255:'Uncertain'};
const key = () => 'roadsense-human-draft-' + plan.frame_plan_sha256;
const status = message => { $('status').textContent = message; };
const current = () => entries[index];
const draft = () => ({schema_version:1, frame_plan_sha256:plan.frame_plan_sha256, samples:entries});
function progress() {
  const reviewed = entries.filter(e => e.human_reviewed).length, complete = entries.filter(e => e.annotation_complete).length;
  $('progress').textContent = `${reviewed} / ${entries.length} reviewed`; $('progress-bar').max = entries.length; $('progress-bar').value = reviewed;
  entries.forEach((e,i) => { $('frame-select').options[i].textContent = `${i+1}. ${e.id} · ${e.human_reviewed?'reviewed':e.annotation_complete?'drawn, pending review':'pending'}`; });
  $('export').textContent = `Export draft · ${complete}/${entries.length} drawn`;
}
function save() {
  dirty = true;
  try { localStorage.setItem(key(), JSON.stringify(draft())); }
  catch { status('Browser autosave failed. Export a draft now.'); }
  progress();
}
function invalidate() { current().annotation_complete = false; current().human_reviewed = false; }
function redraw() {
  const c = $('canvas'), ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  if (!loaded) return;
  ctx.drawImage(image,0,0); const scale = Math.max(1,c.width/800);
  function draw(points,label,closed) {
    if (!points.length) return;
    ctx.beginPath(); ctx.moveTo(...points[0]); points.slice(1).forEach(p => ctx.lineTo(...p));
    if (closed) { ctx.closePath(); ctx.fillStyle = colours[label]+'55'; ctx.fill(); }
    ctx.strokeStyle = colours[label]; ctx.lineWidth = 2*scale; ctx.stroke();
    points.forEach(p => { ctx.beginPath(); ctx.arc(...p,3*scale,0,Math.PI*2); ctx.fillStyle=colours[label]; ctx.fill(); });
  }
  if ($('overlay').checked) current().polygons.forEach(p => draw(p.points,p.label,true));
  draw(pending,Number($('class-select').value),false);
  $('point-count').textContent = `${pending.length} active vertices · ${current().polygons.length} saved polygons`;
}
function controls() {
  const e=current();
  $('negative').checked=e.negative_confirmed; $('complete').checked=e.annotation_complete; $('reviewed').checked=e.human_reviewed;
  $('annotator').value=e.annotator_alias; $('reviewer').value=e.reviewer_alias;
  plan.tags.forEach(t => { $('tag-'+t).value=e.tags[t]; });
  $('review-note').textContent=e.annotator_alias&&e.annotator_alias===e.reviewer_alias?'Same-person review will be recorded as self-reviewed, not independent review.':'A second person is preferable. Same-person review is recorded honestly.';
  $('previous').disabled=index===0; $('next').disabled=index===entries.length-1;
  $('polygon-list').replaceChildren();
  e.polygons.forEach((p,i) => {
    const li=document.createElement('li'); li.textContent=`${names[p.label]} · ${p.points.length} vertices`;
    const remove=document.createElement('button'); remove.textContent='Remove'; remove.setAttribute('aria-label',`Remove polygon ${i+1}`);
    remove.onclick=() => { e.polygons.splice(i,1); invalidate(); save(); controls(); redraw(); };
    li.append(remove); $('polygon-list').append(li);
  }); progress();
}
async function show(i) {
  if (pending.length) { status('Finish or cancel the active polygon before changing frames.'); $('frame-select').value=index; return; }
  index=i; loaded=false; image=null; const row=plan.samples[i], expected=i;
  $('frame-title').textContent=row.id; $('frame-info').textContent=`${i+1}/${entries.length} · ${row.resolution[0]}×${row.resolution[1]} native pixels · ${row.nominal_time_seconds.toFixed(2)}s nominal`;
  $('frame-select').value=i; const c=$('canvas'); c.width=row.resolution[0]; c.height=row.resolution[1];
  $('point-x').max=c.width-1; $('point-y').max=c.height-1;
  controls(); redraw(); status('Loading source frame…'); const incoming=new Image(); incoming.src='/images/'+i;
  try { await incoming.decode(); if(index!==expected)return; if(incoming.naturalWidth!==c.width||incoming.naturalHeight!==c.height)throw Error('Dimensions changed'); image=incoming; loaded=true; redraw(); status('Ready. Draw road, then cut out obstacles. No predictions are loaded.'); }
  catch(error) { if(index===expected)status('Image unavailable: '+error.message); }
}
function addPoint(x,y) {
  if(!loaded)return;
  if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||y<0||x>=image.width||y>=image.height){status('Use in-bounds integer pixel coordinates.');return;}
  if(pending.length>=200){status('Maximum 200 vertices; split into multiple polygons.');return;}
  if(pending.some(p=>p[0]===x&&p[1]===y)){status('Do not repeat a vertex. Finish without repeating the first point.');return;}
  pending.push([x,y]); invalidate(); save(); controls(); redraw();
}
function finish() {
  if(pending.length<3){status('A polygon needs at least three points.');return;}
  if(current().polygons.length>=100){status('Maximum 100 polygons per frame.');return;}
  current().polygons.push({label:Number($('class-select').value),points:pending}); pending=[];
  invalidate(); save(); controls(); redraw(); status('Polygon saved. Non-road cutouts exclude vehicles and other obstacles.');
}
function resume(value) {
  if(value.schema_version!==1||value.frame_plan_sha256!==plan.frame_plan_sha256||!Array.isArray(value.samples)||value.samples.length!==entries.length)throw Error('Wrong frame plan or missing frames');
  const ids=new Set();
  value.samples.forEach((e,i)=>{
    const row=plan.samples[i];
    if(e.id!==row.id||e.image_sha256!==row.image_sha256||ids.has(e.id)||!Array.isArray(e.polygons)||e.polygons.length>100||typeof e.annotation_complete!=='boolean'||typeof e.human_reviewed!=='boolean'||typeof e.negative_confirmed!=='boolean'||typeof e.annotator_alias!=='string'||typeof e.reviewer_alias!=='string')throw Error('Invalid sample');
    ids.add(e.id);
    if(!e.tags||Object.keys(e.tags).length!==plan.tags.length||plan.tags.some(t=>!['present','absent','uncertain'].includes(e.tags[t])))throw Error('Invalid tags');
    e.polygons.forEach(p=>{if(![0,1,255].includes(p.label)||!Number.isInteger(p.label)||!Array.isArray(p.points)||p.points.length<3||p.points.length>200||p.points.some(q=>!Array.isArray(q)||q.length!==2||q.some(n=>!Number.isInteger(n))||q[0]<0||q[1]<0||q[0]>=row.resolution[0]||q[1]>=row.resolution[1]))throw Error('Invalid polygon');});
    if(e.human_reviewed&&(!e.annotation_complete||!e.annotator_alias.trim()||!e.reviewer_alias.trim()))throw Error('Missing review declarations');
  }); entries=value.samples; pending=[];
}
async function initialise() {
  const response=await fetch('/plan.json'); if(!response.ok)throw Error('Cannot load frozen plan'); plan=await response.json();
  entries=plan.samples.map(row=>({id:row.id,image_sha256:row.image_sha256,polygons:[],negative_confirmed:false,annotation_complete:false,human_reviewed:false,annotator_alias:'',reviewer_alias:'',tags:Object.fromEntries(plan.tags.map(t=>[t,'uncertain']))}));
  plan.samples.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=row.id;$('frame-select').append(option);});
  plan.tags.forEach(tag=>{
    const label=document.createElement('label');label.htmlFor='tag-'+tag;label.textContent=tag.replaceAll('_',' ');const select=document.createElement('select');select.id='tag-'+tag;
    ['uncertain','present','absent'].forEach(v=>{const option=document.createElement('option');option.value=v;option.textContent=v;select.append(option);});
    select.onchange=()=>{current().tags[tag]=select.value;current().human_reviewed=false;save();controls();};$('tags').append(label,select);
  });
  let warning='';try{const saved=localStorage.getItem(key());if(saved)resume(JSON.parse(saved));}catch(e){warning='Autosaved draft could not be restored: '+e.message;}
  $('canvas').onclick=e=>{if(!loaded)return;const rect=e.currentTarget.getBoundingClientRect();addPoint(Math.min(image.width-1,Math.floor((e.clientX-rect.left)*e.currentTarget.width/rect.width)),Math.min(image.height-1,Math.floor((e.clientY-rect.top)*e.currentTarget.height/rect.height)));};
  $('canvas').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();finish();}if(e.key==='Backspace'){e.preventDefault();pending.pop();redraw();}if(e.key==='Escape'){pending=[];redraw();}};
  $('finish').onclick=finish;$('undo-point').onclick=()=>{pending.pop();redraw();};$('cancel').onclick=()=>{pending=[];redraw();};$('overlay').onchange=redraw;$('class-select').onchange=redraw;
  $('zoom').onchange=()=>{$('canvas').style.width=(Number($('zoom').value)*100)+'%';};
  $('add-point').onclick=()=>{if(!$('point-x').value||!$('point-y').value){status('Enter both X and Y.');return;}addPoint(Number($('point-x').value),Number($('point-y').value));};
  $('previous').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);$('frame-select').onchange=e=>show(Number(e.target.value));
  ['annotator','reviewer'].forEach(k=>{$(k).onchange=()=>{current()[k+'_alias']=$(k).value.trim();current().human_reviewed=false;save();controls();};});
  $('negative').onchange=()=>{current().negative_confirmed=$('negative').checked;invalidate();save();controls();};
  $('complete').onchange=()=>{if($('complete').checked&&(pending.length||!current().annotator_alias.trim()||(!current().negative_confirmed&&!current().polygons.some(p=>p.label===1)))){status('Finish polygons and enter annotator alias. A no-road frame requires explicit negative confirmation.');controls();return;}current().annotation_complete=$('complete').checked;current().human_reviewed=false;save();controls();};
  $('reviewed').onchange=()=>{if($('reviewed').checked&&(pending.length||!current().annotation_complete||!current().reviewer_alias.trim())){status('Complete annotation and enter the actual reviewer alias first.');controls();return;}current().human_reviewed=$('reviewed').checked;save();controls();};
  $('export').onclick=()=>{if(pending.length){status('Finish or cancel active polygon before exporting.');return;}const url=URL.createObjectURL(new Blob([JSON.stringify(draft(),null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='roadsense_annotations_'+plan.frame_plan_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);dirty=false;status('Draft download requested. Confirm it is saved; no project masks or accuracy generated.');};
  $('import').onchange=async()=>{try{const file=$('import').files[0];if(!file||file.size>20*1024**2)throw Error('Choose JSON smaller than 20 MiB');if(dirty||pending.length){status('Export current work and cancel pending points before replacing the draft.');return;}resume(JSON.parse(await file.text()));save();await show(index);}catch(error){status('Draft not loaded: '+error.message);}};
  window.addEventListener('beforeunload',e=>{if(dirty||pending.length){e.preventDefault();e.returnValue='';}});
  await show(0);if(warning)status(warning+'. No saved data was deleted.');
}
initialise().catch(error=>status('Unable to initialise: '+error.message));

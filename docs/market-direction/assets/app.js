"use strict";
const $ = s => document.querySelector(s);
const fmt = (x, digits=2) => Number.isFinite(x) ? x.toFixed(digits) : "—";
const pct = (x, digits=1) => Number.isFinite(x) ? `${(x*100).toFixed(digits)}%` : "—";
const signed = x => `${x>=0?"+":""}${pct(x)}`;
const esc = x => String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let original, extension, profit;
const state = {dataset:"original",model:"",period:"all",threshold:.55,cost:2};

function finance(r,positions){
  let wealth=1,peak=1,drawdown=0;
  const equity=[100],active=[];
  r.forEach((x,i)=>{wealth*=1+x;peak=Math.max(peak,wealth);drawdown=Math.min(drawdown,wealth/peak-1);equity.push(wealth*100);if(positions[i])active.push(x);});
  const mean=r.reduce((a,b)=>a+b,0)/r.length;
  const sd=Math.sqrt(r.reduce((a,x)=>a+(x-mean)**2,0)/Math.max(1,r.length-1));
  const loss=-active.filter(x=>x<0).reduce((a,b)=>a+b,0);
  return {total_return:wealth-1,annual_return:wealth**(252/r.length)-1,sharpe:sd?mean/sd*Math.sqrt(252):0,max_drawdown:drawdown,exposure:active.length/r.length,round_trips:active.length,hit_rate:active.length?active.filter(x=>x>0).length/active.length:null,profit_factor:loss?active.filter(x=>x>0).reduce((a,b)=>a+b,0)/loss:null,equity};
}
function series(data,probabilities,threshold,cost){
  const position=probabilities.map(x=>x>=threshold?1:0);
  const returns=data.map((x,i)=>position[i]*(x.market_return-2*cost/10000));
  const intraday=data.map(x=>x.market_return-2*cost/10000);
  const hold=data.map(x=>x.buy_hold_return);
  hold[0]-=cost/10000;hold[hold.length-1]-=cost/10000;
  return {position,returns,stats:finance(returns,position),intraday:finance(intraday,data.map(()=>1)),hold:finance(hold,data.map(()=>1))};
}
function kpi(label,value,note){return `<div class="kpi"><span class="kpi-label">${label}</span><strong class="kpi-value">${value}</strong><span class="kpi-note">${note}</span></div>`;}

function lineChart(selector,dates,sets){
  const target=$(selector),W=840,H=275,L=45,R=13,T=15,B=30;
  const all=sets.flatMap(s=>s.values);let lo=Math.min(...all),hi=Math.max(...all);
  const pad=Math.max((hi-lo)*.12,2);lo-=pad;hi+=pad;
  const n=dates.length,xx=i=>L+i/Math.max(1,n-1)*(W-L-R),yy=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Historical equity curves. ${sets.map(s=>s.name).join(', ')}. Hover for daily values."><title>Growth of $100, net of assumed trading costs</title>`;
  for(let i=0;i<5;i++){let value=lo+(hi-lo)*i/4,y=yy(value);svg+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#e8ede8"/><text x="${L-9}" y="${y+3}" text-anchor="end">${fmt(value,0)}</text>`;}
  for(let i=0;i<5;i++){let k=Math.round((n-1)*i/4);svg+=`<text class="axis-label" x="${xx(k)}" y="${H-7}" text-anchor="${i===0?'start':i===4?'end':'middle'}">${dates[k]?.slice(0,7)||''}</text>`;}
  sets.forEach(s=>{svg+=`<path d="${s.values.map((v,i)=>`${i?'L':'M'}${fmt(xx(i),2)},${fmt(yy(v),2)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="${s.name.includes('Long')?'2.6':'1.8'}" stroke-linejoin="round"/>`;});
  svg+=`<line class="hover-line" x1="0" x2="0" y1="${T}" y2="${H-B}" stroke="#7a8c83" stroke-dasharray="3 3" visibility="hidden"/></svg><div class="chart-tooltip" hidden></div>`;
  target.innerHTML=svg;
  const graph=target.querySelector('svg'),tip=target.querySelector('.chart-tooltip'),line=target.querySelector('.hover-line');
  graph.addEventListener('pointermove',event=>{const rect=graph.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*W,k=Math.max(0,Math.min(n-1,Math.round((x-L)/(W-L-R)*(n-1))));line.setAttribute('x1',xx(k));line.setAttribute('x2',xx(k));line.setAttribute('visibility','visible');tip.hidden=false;tip.innerHTML=`${esc(dates[k])}<br>${sets.map(s=>`${esc(s.name)}: $${fmt(s.values[k])}`).join('<br>')}`;});
  graph.addEventListener('pointerleave',()=>{tip.hidden=true;line.setAttribute('visibility','hidden');});
}
function chartSeries(s){return[{name:"Long/cash",color:"#17695e",values:s.stats.equity},{name:"Intraday",color:"#b98133",values:s.intraday.equity},{name:"Buy & hold",color:"#72889d",values:s.hold.equity}];}

function setTab(name,push=true){
  if(!['overview','lab','validation','data'].includes(name))name='overview';
  document.querySelectorAll('.tab-panel').forEach(el=>el.hidden=el.id!==name);
  document.querySelectorAll('[data-tab]').forEach(el=>{if(el.dataset.tab===name)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  if(push)history.replaceState(null,'',`#${name}`);
}
function overall(){
  const r=extension?.report;
  const chosen=original.report.models.find(m=>m.id===original.report.selected_model);
  const c=r?r.classification:chosen.holdout, strategy=r?r.strategy:chosen.strategy;
  const interval=r?.direction_interval;
  $('#overview-period').textContent='HISTORICAL BACKTEST + FRESH EVALUATION';
  $('#overview-kpis').innerHTML=kpi('Net historical return',signed(chosen.strategy.total_return),'2022–2025 · validation-selected model · after costs')+kpi('Fresh balanced accuracy',pct(c.balanced_accuracy),`2026 · 50.0% always-up baseline`)+kpi('Historical net Sharpe',fmt(chosen.strategy.sharpe),'2022–2025 · zero reference rate')+kpi('Historical max drawdown',pct(chosen.strategy.max_drawdown),'2022–2025 · validation-selected model');
  let text=r?`The validation-selected ${r.selected.family} records ${pct(c.accuracy)} accuracy versus ${pct(r.majority.accuracy)} for the always-up baseline on ${r.holdout.rows} fresh sessions. Balanced accuracy is ${pct(c.balanced_accuracy)}. The 95% interval includes 50%, and its trading overlay returns ${signed(strategy.total_return)} after costs.`:`The original validation-selected model records ${pct(c.balanced_accuracy)} balanced accuracy. The evidence does not establish robust directional skill.`;
  $('#finding').innerHTML=`<span aria-hidden="true">→</span><div><h3>${r&&c.balanced_accuracy>.5?'Direction improved. Profitability is a separate test.':'Keep the result separate from the hypothesis.'}</h3><p>${text} The strategy lab includes other models with positive historical net returns; those comparisons retain their actual test status.</p></div>`;
  const observations=original.observations;
  const p=original.probabilities[chosen.id];
  const s=series(observations,p,.55,2);
  lineChart('#overview-chart',[observations[0].signal_date,...observations.map(x=>x.date)],chartSeries(s));
  $('#progression').innerHTML=`<div class="progression-step"><div><b>01 · Original study / 2022–2025</b><p>Fixed specifications, ablations and baselines.</p></div><div class="progression-value">${pct(chosen.holdout.balanced_accuracy)}<small>balanced accuracy · selected model</small></div></div>${r?`<div class="progression-step"><div><b>02 · Enhanced study / 2026</b><p>${r.selected.family}, ${r.selected.window?`${r.selected.window}-year rolling`:'expanding'} training, cross-asset features.</p></div><div class="progression-value">${pct(c.balanced_accuracy)}<small>balanced accuracy · fresh sample</small></div></div>`:''}<div class="progression-step"><div><b>03 · Financial implementation</b><p>Compare net P&L, costs, exposure and benchmarks.</p></div><div class="progression-value"><button class="text-link" data-go="lab">Open strategy lab →</button></div></div>`;
}

function modelOptions(){
  const isFresh=state.dataset==='extension'&&extension;
  if(isFresh){$('#model').innerHTML=`<option value="fresh">${esc(extension.report.selected.family)} · enhanced signals</option><option value="reference">Logistic · original fixed specification</option>`;state.model='fresh';}
  else {state.dataset='original';$('#dataset').value='original';const options=original.report.models.concat(profit?profit.report.models.filter(m=>profit.probabilities[m.id]):[]);$('#model').innerHTML=options.map(m=>`<option value="${m.id}">${esc(m.family)} · ${esc(m.features)}</option>`).join('');state.model=profit?.report.selected_on_validation||original.report.selected_model;$('#model').value=state.model;}
  const observations=isFresh?extension.observations:original.observations;
  const years=[...new Set(observations.map(o=>o.date.slice(0,4)))];$('#period').innerHTML='<option value="all">Full evaluation sample</option>'+years.map(y=>`<option value="${y}">${y}</option>`).join('');state.period='all';
  resetSettings();
}
function resetSettings(){state.threshold=state.dataset==='extension'&&state.model==='fresh'?extension.report.trade_threshold:.55;state.cost=2;$('#threshold').value=state.threshold;$('#cost').value=state.cost;renderLab();}
function dataForState(){
  const data=state.dataset==='extension'?extension.observations:original.observations;
  const allP=state.dataset==='extension'?data.map(o=>state.model==='fresh'?o.probability_up:o.original_probability_up):(original.probabilities[state.model]||profit.probabilities[state.model]);
  const indices=data.map((o,i)=>i).filter(i=>state.period==='all'||data[i].date.startsWith(state.period));
  return {data:indices.map(i=>data[i]),p:indices.map(i=>allP[i])};
}
function renderLab(){
  $('#threshold-value').textContent=pct(state.threshold,0);$('#cost-value').textContent=`${fmt(state.cost,1)} bps`;
  const {data,p}=dataForState(),s=series(data,p,state.threshold,state.cost),v=s.stats;
  $('#lab-kpis').innerHTML=kpi('Net total return',signed(v.total_return),`${data.length} sessions · ${v.round_trips} round trips`)+kpi('Annualized net return',pct(v.annual_return),'Geometric · 252 sessions/year')+kpi('Net Sharpe',fmt(v.sharpe),'Daily returns · zero reference rate')+kpi('Maximum drawdown',pct(v.max_drawdown),'Includes initial capital peak');
  $('#lab-caption').textContent=`${data[0].date} to ${data.at(-1).date} · long when p(up) ≥ ${pct(state.threshold,0)} · ${fmt(state.cost,1)} bps per side`;
  lineChart('#lab-chart',[data[0].signal_date,...data.map(x=>x.date)],chartSeries(s));
  const months={};data.forEach((o,i)=>{const key=o.date.slice(0,7);months[key]=(months[key]??1)*(1+s.returns[i]);});
  const years=[...new Set(data.map(o=>o.date.slice(0,4)))],names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  $('#monthly').innerHTML=`<table class="heatmap"><thead><tr><th>Year</th>${names.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${years.map(year=>`<tr><td class="year">${year}</td>${names.map((_,i)=>{const key=`${year}-${String(i+1).padStart(2,'0')}`,r=months[key]===undefined?null:months[key]-1;return `<td style="background:${r===null?'#f1f3f0':r>=0?`rgba(23,105,94,${Math.min(.38,.08+Math.abs(r)*8)})`:`rgba(163,78,69,${Math.min(.35,.08+Math.abs(r)*8)})`}">${r===null?'—':pct(r,1)}</td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
  $('#exposure').innerHTML=`<div class="exposure-grid"><div><span>Time in market</span><b>${pct(v.exposure)}</b></div><div><span>Winning active days</span><b>${pct(v.hit_rate)}</b></div><div><span>Profit factor</span><b>${fmt(v.profit_factor)}</b></div><div><span>Always-long intraday return</span><b>${signed(s.intraday.total_return)}</b></div><div><span>Buy & hold return</span><b>${signed(s.hold.total_return)}</b></div><div><span>Assumed round-trip cost</span><b>${fmt(2*state.cost,1)} bps</b></div></div><div class="allocation-bar"><span style="width:${v.exposure*100}%"></span></div><p class="footnote">Green: active open-to-close sessions. Remaining sessions are cash. Positive return alone does not imply benchmark outperformance.</p>`;
  $('#ledger').innerHTML=`<table><thead><tr><th>Signal date</th><th>Trade date</th><th class="number">P(up)</th><th>Position</th><th class="number">SPY intraday</th><th class="number">Net P&L</th></tr></thead><tbody>${data.map((o,i)=>({o,i})).slice(-12).reverse().map(({o,i})=>`<tr><td>${o.signal_date}</td><td>${o.date}</td><td class="number">${pct(p[i])}</td><td>${s.position[i]?'Long':'Cash'}</td><td class="number">${signed(o.market_return)}</td><td class="number ${s.returns[i]>=0?'positive':'negative'}">${signed(s.returns[i])}</td></tr>`).join('')}</tbody></table>`;
}

function calibrationChart(points){
  const W=400,H=230,L=40,B=32,T=10,R=18,xx=x=>L+x*(W-L-R),yy=y=>H-B-y*(H-B-T);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reliability plot: mean predicted probability versus observed up rate"><title>Probability calibration</title>`;
  for(const x of [0,.25,.5,.75,1])s+=`<line x1="${xx(0)}" x2="${xx(1)}" y1="${yy(x)}" y2="${yy(x)}" stroke="#e8ede8"/><text x="${L-8}" y="${yy(x)+3}" text-anchor="end">${fmt(x,2)}</text><text x="${xx(x)}" y="${H-12}" text-anchor="middle">${fmt(x,2)}</text>`;
  s+=`<line x1="${xx(0)}" x2="${xx(1)}" y1="${yy(0)}" y2="${yy(1)}" stroke="#a6b4ac" stroke-dasharray="4 4"/>`;
  points.forEach(p=>{s+=`<circle cx="${xx(p.predicted)}" cy="${yy(p.observed)}" r="${Math.max(3,Math.min(10,Math.sqrt(p.count)))}" fill="#17695e" fill-opacity=".75"><title>${p.count} sessions; predicted ${pct(p.predicted)}, observed ${pct(p.observed)}</title></circle>`;});
  $('#calibration').innerHTML=s+'</svg>';
}
function validation(){
  const r=original.report;
  $('#model-table').innerHTML=`<table><thead><tr><th>Model</th><th>Features</th><th class="number">Validation BA</th><th class="number">Holdout BA</th><th class="number">AUC</th><th class="number">Net return</th><th class="number">Net Sharpe</th></tr></thead><tbody>${[...r.models].sort((a,b)=>b.cv_balanced_accuracy-a.cv_balanced_accuracy).map(m=>`<tr class="${m.id===r.selected_model?'selected-row':''}"><td>${esc(m.family)}${m.id===r.selected_model?'<span class="selected-tag">validation pick</span>':''}</td><td>${esc(m.features)}</td><td class="number">${pct(m.cv_balanced_accuracy,2)}</td><td class="number">${pct(m.holdout.balanced_accuracy,2)}</td><td class="number">${fmt(m.holdout.roc_auc,3)}</td><td class="number ${m.strategy.total_return>=0?'positive':'negative'}">${signed(m.strategy.total_return)}</td><td class="number">${fmt(m.strategy.sharpe)}</td></tr>`).join('')}</tbody></table>`;
  $('#ablation').innerHTML=['CatBoost','Random Forest'].map(family=>{const models=r.models.filter(m=>m.family===family),base=models.find(m=>m.features==='Market').cv_balanced_accuracy;return `<p class="eyebrow" style="margin-top:15px">${family}</p>`+models.filter(m=>m.features!=='Market').map(m=>`<div class="ablation-row"><span>${m.features}</span><span class="${m.cv_balanced_accuracy>=base?'positive':'negative'}">${m.cv_balanced_accuracy>=base?'+':''}${fmt((m.cv_balanced_accuracy-base)*100,2)} pp</span></div>`).join('');}).join('')+'<p class="footnote">Negative incremental performance is informative. The strongest fresh extension uses enhanced market features; it does not establish incremental value from sentiment or macro inputs.</p>';
  const e=extension?.report,chosen=r.models.find(m=>m.id===r.selected_model);
  calibrationChart(e?e.calibration:chosen.calibration);
  if(e){
    $('#extension-summary').innerHTML=`<div class="extension-callout"><b>Validation-selected configuration:</b> ${e.selected.family}, ${esc(e.selected.features.toLowerCase())}, depth ${e.selected.depth||'n/a'}, ${e.selected.window||'expanding'}-year history. Classification threshold ${pct(e.selected.threshold,0)}; trading threshold ${pct(e.trade_threshold,0)}. Fresh accuracy: ${pct(e.classification.accuracy)} versus ${pct(e.majority.accuracy)} always-up.</div>`;
    $('#search-table').innerHTML=`<table><thead><tr><th>Development rank</th><th>Model / features</th><th>Window</th><th class="number">Threshold</th><th class="number">Validation BA</th><th class="number">Validation AUC</th></tr></thead><tbody>${[...e.search].sort((a,b)=>b.cv_balanced_accuracy-a.cv_balanced_accuracy).slice(0,8).map((m,i)=>`<tr class="${m.id===e.selected.id?'selected-row':''}"><td>${i+1}${m.id===e.selected.id?'<span class="selected-tag">selected</span>':''}</td><td>${esc(m.family)} / ${esc(m.features)}</td><td>${m.window?m.window+' years':'Expanding'}</td><td class="number">${pct(m.threshold,0)}</td><td class="number">${pct(m.cv_balanced_accuracy,2)}</td><td class="number">${fmt(m.cv_auc,3)}</td></tr>`).join('')}</tbody></table>`;
  }else $('#extension-section').hidden=true;
  const importance=e?e.feature_importance:r.feature_importance.filter(x=>x.model===r.selected_model);
  const sorted=[...importance].sort((a,b)=>b.importance-a.importance).slice(0,9),max=sorted[0].importance;
  $('#importance-note').textContent=e?`${e.selected.family} · fitted on development data`:'Original selected model';
  $('#importance').innerHTML=sorted.map(x=>`<div class="bar-row"><span>${esc(x.feature.replaceAll('_',' '))}</span><div class="bar-track"><div class="bar-fill" style="width:${x.importance/max*100}%"></div></div><span class="bar-value">${pct(x.importance)}</span></div>`).join('');
  const b=e?e.bootstrap:r.bootstrap;
  $('#uncertainty').innerHTML=(e?`<div class="interval"><b>${pct(e.direction_interval.lower_95)} – ${pct(e.direction_interval.upper_95)}</b><span>95% interval for fresh balanced accuracy</span></div><p class="body-copy">The interval includes the 50% baseline. ${e.holdout.rows} sessions are not enough to establish a durable forecasting advantage.</p>`:'')+`<div class="interval"><b>${pct(b.lower_95)} to ${pct(b.upper_95)}</b><span>95% interval · annual arithmetic excess versus intraday baseline</span></div><p class="footnote">${b.repetitions.toLocaleString()} bootstrap draws; ${b.block_sessions}-session circular blocks. The resampling does not remove revised-data bias or correct for all model-selection effects.</p>`;
  if(profit){const parent=$('#model-table').closest('.card');const card=document.createElement('section');card.className='card';card.innerHTML=`<div class="card-heading"><div><h3>Net profitability comparison</h3><p>Historical backtest · 2022–2025 · 55% trade threshold · 2 bps per side</p></div></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Features</th><th class="number">Validation Sharpe</th><th class="number">Net return</th><th class="number">Annual return</th><th class="number">Max drawdown</th></tr></thead><tbody>${[...profit.report.models].sort((a,b)=>b.cv_net_sharpe-a.cv_net_sharpe).map(m=>`<tr><td>${esc(m.family)}${m.id===profit.report.selected_on_validation?'<span class="selected-tag">validation pick</span>':''}</td><td>${esc(m.features)}</td><td class="number">${fmt(m.cv_net_sharpe)}</td><td class="number ${m.strategy.total_return>=0?'positive':'negative'}">${signed(m.strategy.total_return)}</td><td class="number">${pct(m.strategy.annual_return)}</td><td class="number">${pct(m.strategy.max_drawdown)}</td></tr>`).join('')}</tbody></table></div><p class="footnote">Extra Trees and histogram gradient boosting were added after the original sample was inspected. Their results reuse the historical period. The development-selection rule and testing history are preserved in the repository.</p>`;parent.after(card);}
}
function provenance(){
  const manifest=extension?.report.data||original.report.data;
  $('#provenance').innerHTML=`<p class="eyebrow" style="margin-top:22px">SOURCE SNAPSHOT CHECKSUMS · EXTENSION</p>`+Object.entries(manifest.files).map(([name,info])=>`<div class="source-row"><span><b>${esc(name)}</b> · ${info.rows.toLocaleString()} rows</span><code>SHA-256 ${info.sha256.slice(0,16)}…</code></div>`).join('')+`<p class="footnote">Full URLs, UTC retrieval times and file hashes are recorded in provenance.json. Code and original documentation use the MIT license; third-party data retain provider terms.</p>`;
}
async function load(){
  try{
    const response=await fetch('data/research.json');if(!response.ok)throw new Error('Research artifact unavailable');original=await response.json();
    const extras=await Promise.allSettled(['extension','profitability'].map(async name=>{const r=await fetch(`data/${name}.json`);if(!r.ok)throw new Error(name);return r.json();}));
    extension=extras[0].status==='fulfilled'?extras[0].value:null;profit=extras[1].status==='fulfilled'?extras[1].value:null;
    if(!extension){state.dataset='original';$('#dataset option[value=extension]').remove();}
    overall();validation();provenance();modelOptions();
    $('#loading').hidden=true;$('#research').hidden=false;setTab(location.hash.slice(1)||'overview',false);
    document.addEventListener('click',event=>{const button=event.target.closest('[data-tab],[data-go]');if(button){setTab(button.dataset.tab||button.dataset.go);window.scrollTo({top:0,behavior:'smooth'});}});
    window.addEventListener('hashchange',()=>setTab(location.hash.slice(1),false));
    $('#dataset').addEventListener('change',event=>{state.dataset=event.target.value;modelOptions();});
    $('#model').addEventListener('change',event=>{state.model=event.target.value;resetSettings();});
    $('#period').addEventListener('change',event=>{state.period=event.target.value;renderLab();});
    $('#threshold').addEventListener('input',event=>{state.threshold=Number(event.target.value);renderLab();});
    $('#cost').addEventListener('input',event=>{state.cost=Number(event.target.value);renderLab();});
    $('#reset').addEventListener('click',resetSettings);
    // Expose only calculations for reproducible UI checks, never credentials.
    window.researchFinance=finance;
  }catch(error){$('#loading').hidden=true;$('#error').hidden=false;$('#error').textContent=`Unable to load the research artifacts. Refresh this page or open the repository. ${error.message}`;}
}
load();

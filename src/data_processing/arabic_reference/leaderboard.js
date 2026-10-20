'use strict';
const data = JSON.parse(document.getElementById('board-data').textContent);
const node = id => document.getElementById(id);
const make = (tag, text, cls) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; };
const count = n => n.toLocaleString('en-US');
const pct = n => Number.isFinite(n) ? (100 * n).toFixed(2) + '%' : '—';
const displayModel = name => data.model_details?.[name]?.display_name || (['qasr', 'gqasr'].includes(name) ? name.toUpperCase() : name);
const names = {sada:'SADA', commonvoice:'Common Voice', masc_clean:'MASC · Clean', masc_noisy:'MASC · Noisy', mgb2:'MGB-2', casablanca:'Casablanca'};
const palette = ['#74baff', '#e6c27a', '#b4a3ff', '#7ad9e1', '#f5a3c1', '#9fcf9b', '#eaaa80', '#93a9ed', '#d2c1a6'];
const colorOrder = ['qasr', 'gqasr', ...data.models.filter(name => !['qasr', 'gqasr'].includes(name))];
const modelColor = name => palette[colorOrder.indexOf(name) % palette.length];
const option = (select, text, value) => { const el = make('option', text); el.value = value; select.append(el); };
const dot = model => { const el = make('span', undefined, 'dot'); el.style.backgroundColor = modelColor(model); el.setAttribute('aria-hidden', 'true'); return el; };
node('selected').textContent = count(data.selected_clips);
node('hours').textContent = data.duration_hours.toFixed(2);
node('sets').textContent = String(data.datasets.length).padStart(2, '0');
node('models').textContent = String(data.models.length).padStart(2, '0');
node('release-line').textContent = data.dataset_release ? `${data.release_id} / dataset revision ${data.dataset_release.revision.slice(0, 10)}` : data.release_id;
if (data.comparison_protocol) {
  const p = data.comparison_protocol;
  const note = make('p', `Fresh evaluation · ${p.gpus.length} GPUs · ${p.workers_per_gpu} workers/GPU · batch ${p.batch_size} · BF16 · Arabic hint. Jais diacritics ${p.jais_diacritics ? 'enabled' : 'disabled'}.`, 'coverage');
  node('ranking').insertBefore(note, node('ranking').children[1]);
}
if (data.dataset_release) { node('dataset-link').href = data.dataset_release.url; node('dataset-link').hidden = false; }
for (const p of data.panels) option(node('panel'), p.label, p.id);
if (data.panels.some(p => p.id === 'original-full')) node('panel').value = 'original-full';
option(node('dataset'), 'Six-dataset macro', 'macro');
for (const ds of data.datasets) { option(node('dataset'), names[ds] || ds, ds); option(node('example-dataset'), names[ds] || ds, ds); }
for (const model of data.models) { const label = make('span'); label.append(dot(model), make('span', displayModel(model))); node('legend').append(label); }
for (const text of data.limitations) node('limitations').append(make('li', text));
node('draft-note').textContent = data.annotation_source ? `${count(data.annotation_source.eligible_clips)} completed, non-speculative drafts are available. ${count(data.annotation_source.excluded_clips)} clips are excluded from the enhanced comparison.` : 'No enhanced draft bundle has been attached to this release.';
node('review-note').textContent = `${count(data.verified_clips)} of ${count(data.selected_clips)} selected clips have listening attestations. ${data.verified_clips ? 'These are reviewer attestations, not an independent audit.' : 'Audio-reviewed reference scores are not yet available.'}`;
node('provenance').textContent = JSON.stringify({release_id:data.release_id, ranking_source:data.ranking_source, comparison_protocol:data.comparison_protocol, dataset_release:data.dataset_release, annotation_source:data.annotation_source, manifest_digest:data.manifest_digest, policy_sha256:data.policy_sha256, scorer_sha256:data.scorer_sha256, builder_sha256:data.builder_sha256, reviewed_manifest_sha256:data.reviewed_manifest_sha256, prediction_sources:data.prediction_sources, reference_curation_sources:data.reference_curation_sources, model_details:data.model_details, submission_manifest_sha256:data.submission_manifest_sha256}, null, 2);
node('example-sampling').textContent = data.example_sampling + ' Text preview only; listen to the source audio before judging accuracy.';

for (const model of data.models) {
  const perf = data.performance?.[model];
  const tr = make('tr');
  const label = make('td', undefined, 'performance-model');
  const title = make('div', undefined, 'model-name'); title.append(dot(model), make('span', displayModel(model)));
  label.append(title, make('div', perf?.status === 'available' ? `${count(perf.clips)} clips · ${perf.modes.join(' + ')} modes · ${count(perf.outputs)} outputs` : 'Timing files not supplied', 'perf-subtitle')); tr.append(label);
  const available = perf?.status === 'available';
  for (const [value, unit, missing] of [[perf?.mean_batch_seconds, ' s', 'Not recorded'], [perf?.amortized_compute_seconds_per_output, ' s', 'Unavailable'], [perf?.outputs_per_wall_second, '', 'Unavailable']]) {
    const td = make('td', Number.isFinite(value) ? value.toFixed(3) + unit : missing, 'num'); tr.append(td);
  }
  const config = make('td', undefined, 'perf-config');
  if (available) {
    for (const c of perf.configurations) {
      config.append(make('div', `${c.gpu_count} GPUs · ${c.workers_per_gpu} workers/GPU · batch ${c.batch_size}`), make('div', c.devices.join(', '), 'perf-subtitle'));
    }
    config.append(make('div', `${perf.precision.toUpperCase()} · ${perf.attempt_count} attempt(s) · ${perf.total_attempt_wall_seconds.toFixed(1)} s wall · ${perf.oom_batch_splits} OOM splits`, 'perf-subtitle'));
    if (data.model_details?.[model]?.decoding_options?.diacritics !== undefined) {
      config.append(make('div', `Diacritics ${data.model_details[model].decoding_options.diacritics ? 'enabled' : 'disabled'}`, 'perf-subtitle'));
    }
  } else config.textContent = perf?.reason || 'Attach the source run to include timing.';
  tr.append(config); node('performance-results').append(tr);
}

const guide = data.reference_guide;
node('reference-eligible').textContent = count(data.annotation_source?.eligible_clips || 0);
node('reference-coverage-note').textContent = data.annotation_source ? `${count(data.annotation_source.excluded_clips)} clips excluded from enhanced scoring. Original scores in the enhanced comparison use these same eligible clips.` : 'No enhanced annotation bundle is attached to this release.';
node('reference-example-selection').textContent = guide?.selection || 'Editorial examples are not available in this release.';
node('reference-examples-empty').hidden = !!guide?.examples?.length;
for (const example of guide?.examples || []) {
  const card = make('article', undefined, 'card reference-example');
  card.append(make('p', `${names[example.dataset] || example.dataset} · ${example.id}`, 'reference-example-id'), make('h3', example.title));
  for (const [label, text, cls] of [['Original reference', example.original, ''], ['Enhanced draft · ideal_text_draft', example.ideal_draft, ' draft']]) {
    const block = make('div', undefined, 'reference-pair' + cls);
    const content = make('p', text, 'arabic-text'); content.lang = 'ar'; content.dir = 'rtl';
    block.append(make('span', label, 'reference-pair-label'), content); card.append(block);
  }
  const explanation = make('p', undefined, 'reference-explanation');
  for (const part of example.explanation.split(/([\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]+(?: +[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]+)*)/u)) {
    if (/[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]/u.test(part)) { const span = make('bdi', part); span.dir = 'rtl'; span.lang = 'ar'; explanation.append(span); }
    else explanation.append(document.createTextNode(part));
  }
  card.append(explanation, make('p', example.normalized_equal ? 'ELM check: these two texts become identical after normalization. Strict scoring retains their differences.' : 'ELM check: differences remain after normalization, so both normalized and strict scoring can be affected.', 'reference-metric-note'));
  node('reference-examples').append(card);
}
for (const [id, track] of [['explore-enhanced-normalized', 'draft_elm'], ['explore-enhanced-strict', 'draft_strict']]) {
  const panel = data.panels.find(p => p.id === 'enhanced-full') || data.panels.find(p => p.reference_status === 'candidate_assisted_unverified');
  node(id).disabled = !panel;
  node(id).onclick = () => {
    node('panel').value = panel.id; render(true); node('track').value = track; render();
    location.hash = 'ranking';
  };
}

function render(updateTracks = false) {
  const panel = data.panels.find(p => p.id === node('panel').value) || data.panels[0];
  if (!panel) { node('empty').hidden = false; return; }
  if (updateTracks) {
    const normalized = !node('track').value.endsWith('_strict');
    node('track').replaceChildren();
    for (const track of Object.keys(data.tracks).filter(t => panel.scores.some(s => s.track === t))) option(node('track'), data.tracks[track], track);
    const desired = (panel.reference_status === 'candidate_assisted_unverified' ? 'draft' : 'original') + (normalized ? '_elm' : '_strict');
    if ([...node('track').options].some(o => o.value === desired)) node('track').value = desired;
  }
  const track = node('track').value, ds = node('dataset').value, metric = node('metric').value;
  const exploratory = panel.reference_status === 'candidate_assisted_unverified';
  const draft = track.startsWith('draft_');
  node('coverage').classList.toggle('exploratory', exploratory);
  node('coverage').textContent = `${exploratory ? 'EXPLORATORY · ' : ''}${count(panel.scored_clips)} / ${count(panel.selected_clips)} cohort clips · ${panel.full_suite ? 'complete pinned suite' : 'subset'}${panel.cohort === 'diagnostic' ? ' · outcome-selected diagnostics' : ''}. ${exploratory ? 'Model-assisted, unverified drafts. Speculative and incomplete drafts excluded. Original and enhanced tracks use the same clips.' : 'Every listed model uses the same reference track and clips.'}`;
  node('track-note').textContent = track.endsWith('_strict') ? 'Strict scoring retains spelling, punctuation and diacritics. It is whole-transcript agreement, not an isolated formatting-quality score.' : 'Normalized scoring applies the pinned ELM rules to both texts. The six-dataset macro weights each dataset equally; missing rates remain unavailable.';
  node('delta-note').textContent = draft ? `Δ original is the ${metric.toUpperCase()} percentage-point difference on these exact same clips. A lower draft score is not proof of better transcription accuracy.` : 'WER and CER are percentages. No scores have been inferred for missing references.';
  node('delta-heading').textContent = `Δ original ${metric.toUpperCase()}`;
  const rows = panel.scores.filter(r => r.track === track && r.dataset === ds).sort((a,b) => (a[metric] ?? Infinity) - (b[metric] ?? Infinity) || a.model.localeCompare(b.model));
  node('results').replaceChildren(); node('empty').hidden = rows.length > 0;
  let last = null, rank = 0;
  rows.forEach((r, index) => {
    const valid = Number.isFinite(r[metric]);
    if (valid && (last === null || r[metric] !== last)) rank = index + 1;
    last = r[metric];
    const tr = make('tr', undefined, valid && rank === 1 ? 'top-row' : '');
    const rankCell = make('td', undefined, 'rank-cell'); rankCell.append(make('span', valid ? String(rank).padStart(2, '0') : '—', valid && rank === 1 ? 'rank-badge' : '')); tr.append(rankCell);
    const model = make('td', undefined, 'model-cell'); const name = make('div', undefined, 'model-name'); const modelLabel = make(data.model_details?.[r.model]?.model_url ? 'a' : 'span', displayModel(r.model)); if (modelLabel.tagName === 'A') { modelLabel.href = data.model_details[r.model].model_url; modelLabel.target = '_blank'; modelLabel.rel = 'noopener noreferrer'; } name.append(dot(r.model), modelLabel); model.append(name, make('div', 'Arabic language hint', 'model-caption')); tr.append(model);
    for (const key of ['wer', 'cer']) { const td = make('td', undefined, 'num'); td.append(make('span', pct(r[key]), `rate ${metric === key ? 'main-rate' : ''} ${Number.isFinite(r[key]) ? '' : 'unavailable'}`)); tr.append(td); }
    const base = panel.scores.find(s => s.model === r.model && s.dataset === ds && s.track === track.replace(/^draft_/, 'original_'));
    const delta = draft && Number.isFinite(r[metric]) && Number.isFinite(base?.[metric]) ? (r[metric] - base[metric]) * 100 : null;
    tr.append(make('td', delta === null ? '—' : `${delta > 0 ? '+' : ''}${delta.toFixed(2)} pp`, 'num delta'), make('td', count(r.utterances), 'num clip-count'));
    node('results').append(tr);
  });
  node('breakdown').replaceChildren();
  for (const dataset of data.datasets) {
    const scores = panel.scores.filter(r => r.track === track && r.dataset === dataset).sort((a,b) => data.models.indexOf(a.model) - data.models.indexOf(b.model));
    const summary = data.dataset_summary.find(d => d.dataset === dataset);
    const card = make('button', undefined, `dataset-card${ds === dataset ? ' active' : ''}`); card.type = 'button'; card.setAttribute('aria-label', `Show ${names[dataset] || dataset} rankings`); card.setAttribute('aria-pressed', String(ds === dataset));
    const top = make('div', undefined, 'dataset-top'); top.append(make('strong', names[dataset] || dataset), make('span', metric.toUpperCase() + ' ↓'));
    card.append(top, make('div', `${count(scores[0]?.utterances || 0)} scored / ${count(summary.clips)} selected clips`, 'dataset-meta'));
    const max = Math.max(.01, ...scores.map(r => r[metric] || 0));
    for (const score of scores) {
      const row = make('div', undefined, 'bar-row'); const trackEl = make('div', undefined, 'bar-track'); const bar = make('div', undefined, 'bar'); bar.style.backgroundColor = modelColor(score.model); bar.style.width = Number.isFinite(score[metric]) ? `${Math.min(100, 100 * score[metric] / max)}%` : '0'; trackEl.append(bar);
      const modelLabel = make('span', displayModel(score.model)); modelLabel.title = displayModel(score.model); row.append(modelLabel, trackEl, make('strong', pct(score[metric]))); card.append(row);
    }
    if (!scores.length) card.append(make('p', 'No eligible clips', 'small muted'));
    card.onclick = () => { node('dataset').value = dataset; render(); };
    node('breakdown').append(card);
  }
}
let exampleIndex = 0;
function renderExample() {
  const examples = data.examples.filter(e => e.dataset === node('example-dataset').value);
  exampleIndex = Math.max(0, Math.min(exampleIndex, examples.length - 1));
  const sample = examples[exampleIndex];
  node('example-texts').replaceChildren();
  node('example-prev').disabled = exampleIndex === 0; node('example-next').disabled = exampleIndex >= examples.length - 1;
  node('example-count').textContent = examples.length ? `${exampleIndex + 1} / ${examples.length}` : '0 / 0';
  node('example-meta').textContent = sample ? `${sample.id} · ${sample.duration_seconds.toFixed(2)} s${sample.dialect ? ' · ' + sample.dialect : ''}` : 'No transcript examples available';
  node('example-flags').textContent = '';
  if (!sample) return;
  const fields = [['Original reference', 'Source dataset text', sample.original, false], ...Object.entries(sample.predictions).map(([name,text]) => [displayModel(name), 'Model prediction', text, false])];
  if (sample.ideal_draft !== null) fields.push(['Enhanced draft', 'Unverified editorial proposal', sample.ideal_draft, true]);
  for (const [title, subtitle, text, draft] of fields) {
    const row = make('div', undefined, 'text-row' + (draft ? ' draft' : '')); const label = make('div', title, 'text-label'); label.append(make('span', subtitle));
    const content = make('p', text || '(empty transcript)', 'arabic-text'); content.setAttribute('dir', 'auto'); content.setAttribute('lang', 'ar'); row.append(label, content); node('example-texts').append(row);
  }
  const flags = [...sample.flags, ...sample.editorial_flags];
  node('example-flags').textContent = flags.length ? 'Recorded review flags: ' + flags.join(' · ') + '. Flags are review signals, not established errors.' : 'No automatic review flags recorded for this example; this does not certify its accuracy.';
}
node('panel').onchange = () => render(true);
for (const id of ['track', 'dataset', 'metric']) node(id).onchange = () => render();
node('example-dataset').onchange = () => { exampleIndex = 0; renderExample(); };
node('example-prev').onclick = () => { exampleIndex--; renderExample(); };
node('example-next').onclick = () => { exampleIndex++; renderExample(); };
render(true); renderExample();

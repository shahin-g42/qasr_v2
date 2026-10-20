'use strict';
// Uses the shared safe DOM helpers; transcript/model values never enter innerHTML.
{
  const profile = data.latency;
  const reports = profile?.models || {};
  const completed = Object.entries(reports).filter(([, r]) => r.status === 'complete');
  const number = (v, suffix = '') => Number.isFinite(v) ? v.toFixed(3) + suffix : '—';
  const select = node('latency-group');
  option(select, 'All selected clips', 'overall');
  if (profile) {
    const p = profile.protocol;
    node('latency-protocol').textContent = `${p.batch_size} clip/request · ${p.concurrency} concurrent request · ${count(p.clips.length)} ${p.per_dataset ? 'sampled' : 'full-suite'} clips · ${p.repeats} repetitions · ${p.warmup_requests} warm-up requests/model`;
    node('latency-hardware').textContent = `${profile.hardware.identity.name} · ${profile.hardware.identity['memory.total']} MiB · BF16 · 1 CPU thread · PyTorch ${profile.runtime.torch} · Transformers ${profile.runtime.transformers} · ${completed.length} completed model profiles`;
    for (const ds of data.datasets) option(select, names[ds] || ds, `by_dataset:${ds}`);
    for (const group of Object.keys(completed[0]?.[1].summary.by_duration || {})) option(select, `Audio duration · ${group}`, `by_duration:${group}`);
    node('provenance').textContent = JSON.stringify({...JSON.parse(node('provenance').textContent), latency_profile_sha256:profile.profile_sha256}, null, 2);
  }
  if (completed.length) {
    node('latency-empty').hidden = true;
    node('latency-content').hidden = false;
    select.disabled = false;
    for (const [model, report] of completed) {
      const overall = report.summary.overall;
      const card = make('article');
      card.append(make('h4', displayModel(model)),
        make('p', `Model load ${number(report.model_load_seconds, ' s')} · first request ${number(report.summary.first_request_seconds, ' s')}`),
        make('p', `Warm-up mean ${number(report.summary.warmup.request_seconds.mean, ' s')} · token budget ${report.source.max_new_tokens}`),
        make('p', `${overall.generation_limit_hits} token-limit hits · ${overall.empty_outputs} empty outputs · ${overall.refusals} refusals · ${overall.generation_limit_unobservable} calls without token-limit diagnostics`),
        make('p', `Mean input preparation ${number(overall.input_preparation_seconds.mean, ' s')} · inference & decode ${number(overall.inference_decode_seconds.mean, ' s')}`));
      node('latency-startup-results').append(card);
    }
    node('latency-startup-results').append(make('p', 'Model load follows checksum validation and may benefit from the OS file cache. Phase boundaries differ: public adapters preprocess audio inside the inference phase. Compare total request latency across adapters.', 'muted small'));
  }
  function renderLatency() {
    node('latency-results').replaceChildren();
    const [kind, key] = select.value.split(':');
    for (const model of data.models) {
      const report = reports[model];
      const stats = report?.status === 'complete' ? (kind === 'overall' ? report.summary.overall : report.summary[kind]?.[key]) : null;
      const tr = make('tr');
      const cell = make('td', undefined, 'performance-model');
      const label = make('div', undefined, 'model-name');
      label.append(dot(model), make('span', displayModel(model)));
      cell.append(label, make('div', stats ? `${count(stats.unique_clips)} clips · ${count(stats.successful)} successful calls` : 'Profile unavailable for this checkpoint', 'perf-subtitle'));
      tr.append(cell);
      for (const value of ['mean','p50','p95','p99'].map(k => stats?.request_seconds[k]))
        tr.append(make('td', number(value, ' s'), 'num'));
      tr.append(make('td', number(stats?.aggregate_rtf), 'num'),
        make('td', number(stats?.peak_allocated_gib), 'num'),
        make('td', stats ? `${count(stats.attempted)} / ${count(stats.failed)}` : '—', 'num'));
      node('latency-results').append(tr);
    }
  }
  select.onchange = renderLatency;
  renderLatency();
}

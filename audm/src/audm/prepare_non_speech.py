"""Download pinned FSD50K assets, prepare review candidates, and build AUD negatives.

Weak event labels cannot establish absence of speech. Only listened-to, explicitly
accepted candidates become training targets. No model or dataset remote code runs.
"""

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

REPO = "quinnlue/FSD50K-16k"
REVISION = "2a60d475f4e2f0db624a902881af2df79d656145"
SR = 16000
# Process held-out partitions first; exact duplicate audio cannot leak into train.
SPLITS = ("test", "validation", "train")
MUSIC = set(
    """Accordion Acoustic_guitar Bass_drum Bass_guitar Bowed_string_instrument
Brass_instrument Crash_cymbal Cymbal Drum Drum_kit Electric_guitar Glockenspiel Gong
Guitar Harmonica Harp Hi-hat Keyboard_(musical) Mallet_percussion Marimba_and_xylophone
Music Musical_instrument Organ Percussion Piano Plucked_string_instrument
Rattle_(instrument) Scratching_(performance_technique) Snare_drum Strum Tabla Tambourine
Trumpet Wind_instrument_and_woodwind_instrument""".split()
)
NOISE = set(
    """Accelerating_and_revving_and_vroom Aircraft Alarm Animal Bark
Bathtub_(filling_or_washing) Bell Bicycle Bicycle_bell Bird
Bird_vocalization_and_bird_call_and_bird_song Boat_and_Water_vehicle Boiling Boom
Bus Buzz Camera Car Car_passing_by Cat Chicken_and_rooster Chime Chink_and_clink
Chirp_and_tweet Church_bell Clock Coin_(dropping) Computer_keyboard Crack Crackle
Cricket Crow Crumpling_and_crinkling Crushing Cupboard_open_or_close Cutlery_and_silverware
Dishes_and_pots_and_pans Dog Domestic_animals_and_pets Domestic_sounds_and_home_sounds
Door Doorbell Drawer_open_or_close Drill Drip Engine Engine_starting Explosion
Fill_(with_liquid) Fire Fireworks Fixed-wing_aircraft_and_airplane Fowl Frog Frying_(food)
Glass Growling Gull_and_seagull Gunshot_and_gunfire Gurgling Hammer Hiss Idling Insect
Keys_jangling Knock Liquid Livestock_and_farm_animals_and_working_animals Mechanical_fan
Mechanisms Meow Microwave_oven Motor_vehicle_(road) Motorcycle Ocean Packing_tape_and_duct_tape
Pour Power_tool Printer Purr Race_car_and_auto_racing Rail_transport Rain Raindrop
Ratchet_and_pawl Rattle Run Sawing Scissors Screech Shatter Sink_(filling_or_washing)
Siren Skateboard Slam Sliding_door Splash_and_splatter Squeak Stream Subway_and_metro_and_underground
Tap Tearing Thump_and_thud Thunder Thunderstorm Tick Tick-tock Toilet_flush Tools
Traffic_noise_and_roadway_noise Train Trickle_and_dribble Truck Typewriter Typing Vehicle
Vehicle_horn_and_car_horn_and_honking Walk_and_footsteps Water Water_tap_and_faucet Waves_and_surf
Whoosh_and_swoosh_and_swish Wild_animals Wind Wind_chime Wood Writing Zipper_(clothing)""".split()
)
VOCAL = re.compile(
    r"\b(speech|speak\w*|talk\w*|voice\w*|vocal\w*|sing\w*|song|choir|choral|"
    r"conversation|chatter|crowd|cheer\w*|whisper\w*|shout\w*|scream\w*|yell\w*|"
    r"laugh\w*|giggl\w*|chuckl\w*|crying|sobbing|cough\w*|sneez\w*|breath\w*|"
    r"respiratory|sigh|gasp|human|narrat\w*|dialog\w*|rap|lyrics)\b",
    re.I,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path):
    with Path(path).open() as handle:
        for line, raw in enumerate(handle, 1):
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object: {path}:{line}")
                yield row


@contextmanager
def new_directory(output):
    """Keep incomplete corpora invisible; never overwrite an existing corpus."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        yield stage, output
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def permitted_license(value):
    if not isinstance(value, str):
        return False
    url = urlparse(value)
    return (
        url.scheme in ("http", "https")
        and url.netloc.lower() in ("creativecommons.org", "www.creativecommons.org")
        and re.fullmatch(r"/(publicdomain/zero/1\.0|licenses/by/[1-4]\.0)/?", url.path) is not None
        and not url.query
        and not url.fragment
    )


def category(row):
    if not permitted_license(row.get("license")):
        return None, "license"
    labels = row.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) for x in labels):
        return None, "labels"
    tags = row.get("tags") or []
    if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
        return None, "metadata"
    # Deliberately conservative: even 'no vocals' is excluded by this heuristic.
    metadata = " ".join(str(row.get(k) or "") for k in ("title", "description"))
    # Bird_song is an animal label, not a human vocal annotation.
    vocal_labels = [x for x in labels if x != "Bird_vocalization_and_bird_call_and_bird_song"]
    if VOCAL.search(" ".join(vocal_labels + tags + [metadata]).replace("_", " ")):
        return None, "possible_vocals"
    if set(labels) <= MUSIC:
        return "music", None
    if set(labels) <= NOISE:
        return "noise", None
    return None, "mixed_or_unapproved_labels"


def verify_snapshot(snapshot):
    snapshot = Path(snapshot)
    receipt = json.loads((snapshot / "aud_download.json").read_text())
    if receipt != {"repo": REPO, "revision": REVISION}:
        raise ValueError("Snapshot source/revision does not match the pinned source")
    entries = json.loads((snapshot / "checksums.json").read_text())
    paths = set()
    for entry in entries:
        relative = entry["path"]
        if not re.fullmatch(r"data/(train|validation|test)-\d+-of-\d+\.parquet", relative):
            raise ValueError(f"Unexpected shard path: {relative}")
        path = snapshot / relative
        if relative in paths or not path.is_file():
            raise ValueError(f"Missing/duplicate shard: {relative}")
        paths.add(relative)
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
    if not paths or any(not any(p.startswith(f"data/{s}-") for p in paths) for s in SPLITS):
        raise ValueError("Snapshot must contain all three source splits")
    if paths != {str(p.relative_to(snapshot)) for p in (snapshot / "data").glob("*.parquet")}:
        raise ValueError("Unexpected unverified parquet file in snapshot")
    return receipt


def download(root, workers=4):
    from huggingface_hub import snapshot_download

    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    snapshot = Path(root).resolve() / "raw" / "FSD50K-16k" / REVISION
    receipt = snapshot / "aud_download.json"
    expected = {"repo": REPO, "revision": REVISION}
    if receipt.exists() and json.loads(receipt.read_text()) != expected:
        raise ValueError("Existing snapshot has a different source receipt")
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(snapshot),
        allow_patterns=[
            "data/*.parquet",
            "checksums.json",
            "README.md",
            "LICENSE-DATASET.md",
            "metadata/**",
            "conversion_report.json",
        ],
        max_workers=workers,
    )
    dump(receipt, expected)
    verify_snapshot(snapshot)
    return {"snapshot": str(snapshot), **expected, "shard_checksums": "verified"}


def parquet_rows(snapshot):
    import pyarrow.parquet as pq

    for split in SPLITS:
        for path in sorted((Path(snapshot) / "data").glob(f"{split}-*.parquet")):
            print(f"Reading {path.name}", flush=True)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16, use_threads=False):
                for row in batch.to_pylist():
                    yield split, row


def read_audio(row):
    encoded = row.get("audio")
    if not isinstance(encoded, dict) or not isinstance(encoded.get("bytes"), bytes):
        raise ValueError("Expected embedded audio bytes; external paths are not opened")
    with sf.SoundFile(io.BytesIO(encoded["bytes"])) as handle:
        if handle.samplerate != SR or handle.channels != 1 or not 0 < len(handle) <= 600 * SR:
            raise ValueError("Expected mono 16 kHz audio, at most 600 seconds")
        wave = handle.read(dtype="float32")
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1:
        raise ValueError("Nonfinite/out-of-range audio")
    if np.max(np.abs(wave)) < 1e-4:
        raise ValueError("Near-silent source needs separate labeling")
    return wave


def review_page(path, candidates):
    # No source text is inserted as HTML, and '<' cannot terminate the JSON script.
    payload = json.dumps(candidates, ensure_ascii=False).replace("<", "\\u003c")
    Path(path).write_text(
        """<!doctype html><meta charset="utf-8"><title>AUD negative review</title>
<style>body{font:16px system-ui;max-width:950px;margin:30px auto}article{border-top:1px solid #bbb;
padding:12px}button,select{padding:8px;margin:5px}audio{display:block;width:100%}</style>
<h1>AUD non-speech review</h1><p>Listen to the entire clip. Accept only the proposed category
with no human speech, singing or uncertain vocals. Keep uncertain clips pending or reject.
Review all three splits. Decisions are kept in this tab; export before closing or refreshing.</p>
<button id="prev">Previous</button><button id="next">Next</button>
<button id="save">Download decisions.jsonl</button><span id="status"></span><main></main>
<script type="application/json" id="data">"""
        + payload
        + """</script><script>
const rows=JSON.parse(document.querySelector('#data').textContent), decisions={}; let page=0;
function render(){const main=document.querySelector('main');main.replaceChildren();
document.querySelector('#status').textContent=`Page ${page+1}/${Math.max(1,Math.ceil(rows.length/20))};
${Object.keys(decisions).length} decisions / ${rows.length} clips`;
for(const r of rows.slice(page*20,page*20+20)){const el=document.createElement('article');
const p=document.createElement('p');p.textContent=`${r.candidate_id} | ${r.split} | ${r.audio_type}
| ${r.duration.toFixed(2)}s | ${r.labels.join(', ')} | ${r.title||''}`;el.append(p);
const a=document.createElement('audio');a.controls=true;a.preload='none';a.src=r.review_audio;el.append(a);
const s=document.createElement('select');for(const v of ['pending','accept','reject']){
const o=document.createElement('option');o.value=v;o.textContent=v;s.append(o)}
s.value=decisions[r.candidate_id]||'pending';s.onchange=()=>{if(s.value==='pending')delete decisions[r.candidate_id];
else decisions[r.candidate_id]=s.value};el.append(s);main.append(el)}}
document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render()};
document.querySelector('#next').onclick=()=>{page=Math.min(Math.max(0,Math.ceil(rows.length/20)-1),page+1);render()};
document.querySelector('#save').onclick=()=>{const text=Object.entries(decisions).map(([candidate_id,decision])=>
JSON.stringify({candidate_id,decision})).join('\\n')+'\\n';const url=URL.createObjectURL(new Blob([text],{type:'application/x-ndjson'}));
const a=document.createElement('a');a.href=url;a.download='decisions.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();</script>"""
    )


def prepare(
    root, output, min_seconds=1.0, max_seconds=30.0, train_per_class=2000, eval_per_class=200
):
    if not (
        math.isfinite(min_seconds)
        and math.isfinite(max_seconds)
        and 0 < min_seconds <= max_seconds <= 30
    ):
        raise ValueError("Require 0 < min_seconds <= max_seconds <= 30")
    if min(train_per_class, eval_per_class) < 1:
        raise ValueError("Per-class candidate limits must be positive")
    min_frames, max_frames = math.ceil(min_seconds * SR), math.floor(max_seconds * SR)
    if max_frames < min_frames:
        raise ValueError("Duration bounds contain no whole audio samples")
    snapshot = Path(root).resolve() / "raw" / "FSD50K-16k" / REVISION
    verify_snapshot(snapshot)
    counts, candidates, sources, hashes = Counter(), [], set(), set()
    with new_directory(output) as (stage, destination):
        for split, row in parquet_rows(snapshot):
            counts["source_records"] += 1
            kind, reason = category(row)
            if reason:
                counts[f"excluded:{reason}"] += 1
                continue
            limit = train_per_class if split == "train" else eval_per_class
            if counts[f"{split}:{kind}"] >= limit:
                counts["excluded:candidate_limit"] += 1
                continue
            identity = row.get("freesound_id")
            if type(identity) is not int or identity <= 0:
                raise ValueError("Expected a positive Freesound ID")
            if identity in sources:
                counts["excluded:duplicate_source"] += 1
                continue
            try:
                wave = read_audio(row)
            except (ValueError, RuntimeError) as exc:
                counts[f"excluded:audio:{type(exc).__name__}"] += 1
                continue
            source_hash = hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest()
            sources.add(identity)
            if source_hash in hashes:
                counts["excluded:duplicate_audio"] += 1
                continue
            # Also record full-source hashes, to catch duplicates with different IDs.
            hashes.add(source_hash)
            for start in range(0, len(wave), max_frames):
                chunk = wave[start : start + max_frames]
                if len(chunk) < min_frames:
                    counts["excluded:short_tail"] += 1
                    continue
                if counts[f"{split}:{kind}"] >= limit:
                    break
                chunk_hash = hashlib.sha256(chunk.astype("<f4").tobytes()).hexdigest()
                if chunk_hash in hashes and chunk_hash != source_hash:
                    counts["excluded:duplicate_chunk"] += 1
                    continue
                if np.max(np.abs(chunk)) < 1e-4:
                    counts["excluded:quiet_chunk"] += 1
                    continue
                hashes.add(chunk_hash)
                candidate_id = f"fsd-{identity}-{start}"
                relative = f"audio/{split}/{candidate_id}.wav"
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, chunk, SR, subtype="PCM_16")
                candidate = {
                    k: row.get(k)
                    for k in (
                        "labels",
                        "mids",
                        "title",
                        "description",
                        "tags",
                        "uploader",
                        "license",
                        "freesound_id",
                        "freesound_url",
                        "source_partition",
                    )
                }
                candidate.update(
                    candidate_id=candidate_id,
                    split=split,
                    audio_type=kind,
                    audio_filepath=str(destination / relative),
                    review_audio=relative,
                    duration=len(chunk) / SR,
                    text=None,
                    review_status="pending",
                    source_repo=REPO,
                    source_revision=REVISION,
                    source_pcm_sha256=source_hash,
                    start_sample=start,
                    end_sample=start + len(chunk),
                    audio_sha256=sha256(path),
                )
                candidates.append(candidate)
                counts[f"{split}:{kind}"] += 1
        if not candidates:
            raise ValueError("No candidates survived the filters")
        with (stage / "candidates.jsonl").open("w") as handle:
            for candidate in candidates:
                handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        review_page(stage / "review.html", candidates)
        report = dict(
            source_repo=REPO,
            source_revision=REVISION,
            counts=dict(counts),
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            train_per_class=train_per_class,
            eval_per_class=eval_per_class,
            candidate_manifest_sha256=sha256(stage / "candidates.jsonl"),
            status="Candidates only: listening review required; text=null blocks training",
        )
        dump(stage / "report.json", report)
    return report


def finalize(candidates, decisions, output, silence_per_split=100, seed=1729):
    if silence_per_split < 1:
        raise ValueError("silence_per_split must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    candidate_path, decision_path = Path(candidates).resolve(), Path(decisions).resolve()
    report = json.loads(candidate_path.with_name("report.json").read_text())
    if sha256(candidate_path) != report["candidate_manifest_sha256"]:
        raise ValueError("Candidate manifest changed since preparation")
    indexed = {}
    for row in jsonl(candidate_path):
        key = row["candidate_id"]
        if key in indexed:
            raise ValueError(f"Duplicate candidate ID: {key}")
        indexed[key] = row
    chosen, seen = [], set()
    for decision in jsonl(decision_path):
        key = decision.get("candidate_id")
        if (
            set(decision) != {"candidate_id", "decision"}
            or not isinstance(key, str)
            or key not in indexed
            or key in seen
            or decision["decision"] not in ("accept", "reject")
        ):
            raise ValueError("Unknown/duplicate candidate or invalid decision")
        seen.add(key)
        if decision["decision"] == "accept":
            chosen.append(indexed[key])
    counts = Counter(f"{r['split']}:{r['audio_type']}" for r in chosen)
    if any(counts[f"{s}:{k}"] == 0 for s in SPLITS for k in ("music", "noise")):
        raise ValueError("Accept at least one music and noise candidate in each source split")
    with new_directory(output) as (stage, destination):
        manifests = {s: [] for s in SPLITS}
        for row in chosen:
            path = Path(row["audio_filepath"])
            if sha256(path) != row["audio_sha256"]:
                raise ValueError(f"Candidate audio changed: {path}")
            # Copy rather than link: future edits to review assets must not mutate training audio.
            relative = Path("audio") / row["split"] / path.name
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha256(target) != row["audio_sha256"]:
                raise ValueError(f"Candidate audio changed during copy: {path}")
            record = {k: v for k, v in row.items() if k != "review_audio"}
            record.update(
                audio_filepath=str(destination / relative), text="", review_status="accepted"
            )
            manifests[row["split"]].append(record)
        for split_index, split in enumerate(SPLITS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_index]))
            for index in range(silence_per_split):
                frames = int(
                    rng.integers(
                        math.ceil(report["min_seconds"] * SR),
                        math.floor(report["max_seconds"] * SR) + 1,
                    )
                )
                # Digital silence is explicit synthetic supervision, not mislabeled HF audio.
                relative = Path("audio") / split / f"silence-{index:06d}.wav"
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.zeros(frames, dtype=np.float32), SR, subtype="PCM_16")
                manifests[split].append(
                    dict(
                        audio_filepath=str(destination / relative),
                        duration=frames / SR,
                        text="",
                        audio_type="silence",
                        split=split,
                        source="synthetic_digital_silence",
                        seed=seed,
                        audio_sha256=sha256(path),
                        review_status="synthetic",
                    )
                )
                counts[f"{split}:silence"] += 1
            # Deterministic interleaving prevents first-N evaluation being all one category.
            rng.shuffle(manifests[split])
            with (stage / f"{split}.jsonl").open("w") as handle:
                for row in manifests[split]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = dict(
            source_repo=REPO,
            source_revision=REVISION,
            counts=dict(counts),
            seed=seed,
            candidates_sha256=sha256(candidate_path),
            decisions_sha256=sha256(decision_path),
            manifests={s: sha256(stage / f"{s}.jsonl") for s in SPLITS},
            silence="Synthetic digital zeros; not a real-room silence benchmark",
            scope="Non-speech only; add disjoint speech examples for four-class evaluation",
        )
        dump(stage / "report.json", result)
        shutil.copyfile(decision_path, stage / "decisions.jsonl")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("download", help="Resume pinned Hub downloads and verify every shard")
    get.add_argument("--root", required=True)
    get.add_argument("--workers", type=int, default=4)
    prep = sub.add_parser("prepare", help="Write candidates, audio and a browser listening page")
    prep.add_argument("--root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--min-seconds", type=float, default=1)
    prep.add_argument("--max-seconds", type=float, default=30)
    prep.add_argument("--train-per-class", type=int, default=2000)
    prep.add_argument("--eval-per-class", type=int, default=200)
    final = sub.add_parser(
        "finalize", help="Build immutable training manifests from listening decisions"
    )
    final.add_argument("--candidates", required=True)
    final.add_argument("--decisions", required=True)
    final.add_argument("--output", required=True)
    final.add_argument("--silence-per-split", type=int, default=100)
    final.add_argument("--seed", type=int, default=1729)
    values = vars(parser.parse_args())
    command = values.pop("command")
    print(
        json.dumps(
            {"download": download, "prepare": prepare, "finalize": finalize}[command](**values),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

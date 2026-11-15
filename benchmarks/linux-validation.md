# Linux validation

RTX 5090 results on the hardware recorded in [transfer-baseline.md](transfer-baseline.md).
Full conditions, tokens, timings and recipes are in [linux-validation.json](linux-validation.json).
Generation, full tests and publication QA completed.

## Comparison scope

The three R2VA requests below match retained prompt/materials, reference bytes/order,
seed, image short edge and sampling parameters. The machines and operating systems differ;
historical cache state is unknown. These are whole-pipeline comparisons, not isolated kernel speedups.

R2VA 5s/10s canonical LF text hashes match their recorded historical text digests.
R2VA 15s matches the retained prompt file and argv, but an independent historical recorded
prompt digest is unavailable. Digest evidence strength is recorded separately in JSON.

| Input video + images | Output | Windows RTX 5080 s | Linux RTX 5090 s | Windows/Linux time |
| --- | --- | ---: | ---: | ---: |
| 5s video + 1 image (short edge 512) | 5s, 1344x768 | 496.331 | 286.746 | 1.73x |
| 10s video + 1 image (short edge 512) | 10s, 1344x768 | 1344.252 | 739.713 | 1.82x |
| 15s video + 1 image (short edge 512) | 15s, 1344x768 | 2258.036 | 1260.393 | 1.79x |

Other cases use new OpenH3-IR prompts, a new dependent chain, or an unmatched canvas.
They demonstrate the task/scale; they do not recover missing historical Windows recipes.

## Generation and memory

Times include the original native export but exclude PE and later H265 publication.
RSS and NVIDIA board memory are sampled observations; CUDA allocation is the allocator-reported peak.

| Case | Input images / video s | Requested / aligned / published frames | Pipeline s | RSS GiB | CUDA allocated GiB | Board sampled GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| chain-second-15s | 2 / 0 | 362 / 362 / 360 | 404.689 | 17.05 | 4.83 | 7.79 |
| f2va-10s-scale | 1 / 0 | 240 / 243 / 240 | 227.641 | 17.08 | 3.69 | 6.02 |
| f2va-15s-scale | 1 / 0 | 360 / 362 / 360 | 355.680 | 16.44 | 4.74 | 7.58 |
| f2va-5s-scale | 1 / 0 | 120 / 124 / 120 | 119.147 | 17.10 | 3.65 | 5.20 |
| fl2va-omni-10s-scale | 2 / 0 | 243 / 243 / 240 | 259.756 | 17.06 | 3.70 | 6.18 |
| fl2va-omni-15s-scale | 2 / 0 | 362 / 362 / 360 | 405.575 | 17.06 | 4.84 | 7.68 |
| fl2va-omni-5s-scale | 2 / 0 | 124 / 124 / 120 | 139.045 | 16.61 | 3.67 | 5.18 |
| fl2va-t2va-10s-scale | 2 / 0 | 243 / 243 / 240 | 326.225 | 17.06 | 3.99 | 7.06 |
| fl2va-t2va-15s-scale | 2 / 0 | 362 / 362 / 360 | 512.944 | 17.07 | 5.28 | 7.62 |
| fl2va-t2va-5s-scale | 2 / 0 | 124 / 124 / 120 | 174.403 | 17.04 | 3.67 | 5.24 |
| r2va-10s | 1 / 10 | 240 / 243 / 240 | 739.713 | 17.92 | 6.64 | 9.57 |
| r2va-15s | 1 / 15 | 360 / 362 / 360 | 1260.393 | 18.10 | 8.49 | 9.88 |
| r2va-480p-5s-scale | 1 / 5 | 120 / 124 / 120 | 111.980 | 17.29 | 3.56 | 4.74 |
| r2va-5s | 1 / 5 | 120 / 124 / 120 | 286.746 | 17.52 | 3.73 | 7.08 |
| t2va-10s-scale | 0 / 0 | 240 / 243 / 240 | 279.216 | 8.74 | 3.80 | 5.85 |
| t2va-15s-scale | 0 / 0 | 360 / 362 / 360 | 435.873 | 9.49 | 5.08 | 7.40 |
| t2va-480p-5s-scale | 0 / 0 | 120 / 124 / 120 | 103.474 | 8.67 | 3.52 | 4.76 |
| t2va-5s-pe | 0 / 0 | 120 / 124 / 120 | 181.366 | 8.35 | 3.64 | 5.23 |

Linux process-tree RSS is not private commit. Container `anon`, `file`, raw current,
inactive-file-adjusted working set and host-visible physical usage remain separate in JSON.
The container working set still includes active file cache; it must not be labeled Python private RAM.
Independent column maxima need not occur simultaneously. Initial runs did not sample every cgroup field.
The container limit was 92 GiB; these runs do not establish performance under a 32 GiB whole-host limit.

The allocator ceiling is 10 GiB, DiT whole-board target 11.5 GiB, minimum free-board reserve 256 MiB,
and additional DiT margin 128 MiB. Actual per-run caps and policies are recorded in each recipe.

## Phase timings

| Case | Preprocess s | Qwen s | Reference VAE s | DiT steps s | Decode s | Native export/finalize s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain-second-15s | 0.934 | 5.465 | 2.196 | 357.569 | 32.674 | 5.523 |
| f2va-10s-scale | 0.900 | 5.177 | 1.999 | 189.477 | 25.278 | 4.521 |
| f2va-15s-scale | 0.943 | 5.285 | 2.001 | 308.967 | 32.733 | 5.455 |
| f2va-5s-scale | 0.932 | 5.212 | 1.983 | 93.206 | 14.167 | 3.381 |
| fl2va-omni-10s-scale | 1.008 | 5.768 | 2.367 | 221.988 | 23.799 | 4.532 |
| fl2va-omni-15s-scale | 0.998 | 5.906 | 2.252 | 357.458 | 33.032 | 5.601 |
| fl2va-omni-5s-scale | 1.013 | 9.681 | 3.150 | 106.316 | 14.535 | 4.075 |
| fl2va-t2va-10s-scale | 1.012 | 5.959 | 2.368 | 287.699 | 23.728 | 5.149 |
| fl2va-t2va-15s-scale | 0.992 | 5.972 | 2.531 | 464.121 | 32.902 | 6.080 |
| fl2va-t2va-5s-scale | 0.971 | 6.228 | 2.608 | 146.283 | 14.268 | 3.785 |
| r2va-10s | 2.722 | 9.863 | 38.306 | 659.669 | 24.135 | 4.696 |
| r2va-15s | 2.636 | 13.468 | 58.474 | 1145.516 | 33.827 | 6.102 |
| r2va-480p-5s-scale | 2.814 | 5.901 | 9.972 | 80.771 | 9.097 | 3.124 |
| r2va-5s | 1.827 | 9.747 | 21.339 | 235.020 | 14.802 | 3.681 |
| t2va-10s-scale | 0.862 | 5.205 | 0.236 | 243.673 | 24.474 | 4.438 |
| t2va-15s-scale | 0.909 | 7.942 | 0.138 | 385.624 | 35.392 | 5.529 |
| t2va-480p-5s-scale | 0.835 | 5.317 | 0.136 | 84.593 | 9.221 | 3.063 |
| t2va-5s-pe | 0.885 | 8.056 | 0.224 | 151.714 | 16.460 | 3.731 |

First-step latency can include shape-dependent initialization and compilation. Compile time was not
isolated; caches were not globally flushed. The three small CUDA installation checks do not establish
cold-model latency. Background downloads and per-step timings are retained for each request.
Offload reads, transfers and compute overlap, so their counters cannot be summed as exclusive wall time.

## DiT timeline and transfer baseline

A separate 15s-input/15s-output prefix profile measured 12 real blocks (1 through 12) in the first
forward, after block 0 and a complete independent identical-prefix warmup. It excludes preprocessing,
final output and the rest of the full eight-forward pipeline. The 226,195 rows comprise 211,056 video,
13,933 Qwen text-stream rows (including visual embeddings) and 1,206 audio rows.

The profiled window was 34.221s versus 33.838s warmup, an observed wall-time difference of about 1.13%.
This single pair estimates profiling overhead and also includes run variance. CUDA activity
union was 99.721%, kernel union 99.379%, the longest CUDA gap 7.55ms, and the worst one-second
window 98.732% active. These timeline fractions do not measure Tensor Core throughput or SM occupancy.
Four actual weight staging slots of 128 MiB each were pinned, with single weight arena, GPU MXFP8
attention outputs, 2048-token QKV chunks and no input quantization cache; attention host-drain gap
attribution was zero. Threshold counts and mutually exclusive gap attribution are in JSON.

The independent [transfer baseline](transfer-baseline.md) reached about 56–57 GB/s for large pinned
copies on both GPUs, under Gen5 x16 load. Its slower pageable Linux copy is a different host-memory
path and cannot be used to explain bubbles in this pinned-transfer profile.

## Tokens and prompt enhancement

| Case | OpenH3 receipt prompt_tokens | Actual Qwen text-stream rows | Video rows / conditioning video rows | Audio rows |
| --- | ---: | ---: | --- | ---: |
| chain-second-15s | 679 | 2707 | 109872 / 2016 | 1206 |
| f2va-10s-scale | 698 | 1711 | 73584 / 1008 | 810 |
| f2va-15s-scale | 620 | 1632 | 108864 / 1008 | 1206 |
| f2va-5s-scale | 502 | 1516 | 38304 / 1008 | 414 |
| fl2va-omni-10s-scale | 718 | 2747 | 74592 / 2016 | 810 |
| fl2va-omni-15s-scale | 734 | 2763 | 109872 / 2016 | 1206 |
| fl2va-omni-5s-scale | 722 | 2752 | 39312 / 2016 | 414 |
| fl2va-t2va-10s-scale | 718 | 2747 | 74592 / 2016 | 810 |
| fl2va-t2va-15s-scale | 734 | 2763 | 109872 / 2016 | 1206 |
| fl2va-t2va-5s-scale | 722 | 2752 | 39312 / 2016 | 414 |
| r2va-10s | not retained | 12605 | 140496 / 67920 | 810 |
| r2va-15s | not retained | 13933 | 211056 / 103200 | 1206 |
| r2va-480p-5s-scale | not retained | 4194 | 27294 / 12864 | 414 |
| r2va-5s | not retained | 7284 | 69936 / 32640 | 414 |
| t2va-10s-scale | 472 | 469 | 72576 / 0 | 810 |
| t2va-15s-scale | 601 | 597 | 107856 / 0 | 1206 |
| t2va-480p-5s-scale | 546 | 545 | 14430 / 0 | 414 |
| t2va-5s-pe | 546 | 545 | 37296 / 0 | 414 |

The OpenH3 field is reproduced as emitted; it is not the actual Qwen embedding length or API billing.
The Qwen text stream can include visual reference tokens. Compiled prompt SHA, seed and ordered input
SHA/shape/duration are recorded in recipes. Selected new prompts use OpenH3-IR 0.4.1 and DeepSeek Flash,
thinking and vision enabled (T2VA has zero visual samples). PE attempts and wall time are separate.
Strict R2VA reuses retained prompts without calling PE during generation.

Published prompts use UTF-8 LF. Recipes retain `original_runtime_raw_bytes_sha256` and
`published_LF_utf8_sha256` / `canonical_text_sha256`; the runtime text reader normalizes CRLF to LF.
No assertion is made that the original CRLF file bytes equal the published LF bytes.

## Frame control and publication

FL2VA requests 124/243/362 aligned frames and publishes 120/240/360 frames at 24 fps.
The 362-frame and 480p cases use local benchmark overrides; the public CLI remains unchanged.
Publication preserves first/last endpoints by affine PTS scaling and fps conversion. Ordinary T/F/R
video speed remains unchanged. Audio is copied when its valid presentation is already exact;
otherwise it is trimmed/padded. Only video-retimed FL segments receive the corresponding audio tempo.
Native tensor lengths, actual decoded media lengths and published lengths are distinct fields.

The chain reuses the actual F2VA first segment and its decoded final frame 359 as the second
segment first-frame reference. Each published segment has 360 frames; concatenation has 720 frames
at 24 fps (30 seconds). Sum requested is 722 and sum aligned is 724; neither is a published-media count.
The second segment is documented but not published redundantly as a separate MP4.

The orange-ball request is an owned-reference functional smoke test, not a matched Windows speed test.
It completed eight forwards in 118.421s, with one image (short edge 256),
no PE, first-use CUDA shapes, populated weight page cache and background T2VA downloads. It is not
comparable with the longer-video reference requests above. Its exact prompt and owned reference SHA
are recorded separately.
All distributed clips follow the credits and usage terms in [the demo notice](../examples/demos/README.md).
Original reference media are retained locally and are not distributed with these recipes.
Use your own assets to reproduce scale; SHA fields identify the recorded requests.

## Tests

- CPU invocation: 403 passed, 51 skipped, 0 failures, 0 errors; JUnit suite 4.685s.
- GPU invocation: 454 passed, 0 skipped, 0 failures, 0 errors; JUnit suite 39.619s.

Fresh virtual-environment bootstrap: complete, 36.297s total using existing caches.
Package compatibility and CPU/GPU doctor checks passed. Three selected CUDA checks passed in
14.48s of pytest time; wrapper wall time is recorded separately.
This verifies a new environment from the shared installer; it is not a cold network-download speed claim.
The four recorded runtime/config/test source files matched the published source snapshot by SHA256.

## Media validation

Generation ran on Linux RTX 5090. Publication used CPU libx265 on the independent Windows machine,
without Torch/CUDA, NVENC or a Windows model rerun; it did not contend with remote GPU measurements.
All published files use H265 CRF20, hvc1, yuv420p, AAC stereo 32 kHz and faststart. Only generated
clips are distributed: 17 core outputs, one 30-second concatenation and the orange smoke test.
The JSON `publications` object is the media QA manifest, including SHA256, byte counts, decoded
frame counts, audio presentation/decoded sample counts, actual filters, encode wall time and
first/last frame PSNR against the original generated video. No raw reference clips, QA images or
original H264 outputs are included in Git.

| File | Frames | Video s | Effective audio s | MiB | Encode s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r2va-5s.mp4` | 120 | 5.000 | 5.000 | 3.59 | 2.163 |
| `r2va-10s.mp4` | 240 | 10.000 | 10.000 | 5.50 | 3.185 |
| `r2va-15s.mp4` | 360 | 15.000 | 15.000 | 6.96 | 5.839 |
| `t2va-5s.mp4` | 120 | 5.000 | 5.000 | 1.50 | 1.535 |
| `f2va-5s.mp4` | 120 | 5.000 | 5.000 | 3.28 | 1.676 |
| `f2va-10s.mp4` | 240 | 10.000 | 10.000 | 5.60 | 2.989 |
| `f2va-15s.mp4` | 360 | 15.000 | 15.000 | 7.57 | 4.788 |
| `fl2va-omni-5s.mp4` | 120 | 5.000 | 5.000 | 3.90 | 1.937 |
| `fl2va-omni-10s.mp4` | 240 | 10.000 | 10.000 | 6.72 | 3.337 |
| `fl2va-omni-15s.mp4` | 360 | 15.000 | 15.000 | 9.74 | 5.594 |
| `r2va-480p-5s.mp4` | 120 | 5.000 | 5.000 | 2.02 | 1.050 |
| `orange-r2va-5s.mp4` | 120 | 5.000 | 5.000 | 0.69 | 1.291 |
| `chain-30s.mp4` | 720 | 30.000 | 30.000 | 15.97 | 11.202 |
| `t2va-10s.mp4` | 240 | 10.000 | 10.000 | 3.28 | 2.917 |
| `t2va-15s.mp4` | 360 | 15.000 | 15.000 | 4.34 | 4.725 |
| `fl2va-t2va-5s.mp4` | 120 | 5.000 | 5.000 | 5.14 | 2.391 |
| `fl2va-t2va-10s.mp4` | 240 | 10.000 | 10.000 | 9.55 | 4.926 |
| `fl2va-t2va-15s.mp4` | 360 | 15.000 | 15.000 | 12.82 | 7.671 |
| `t2va-480p-5s.mp4` | 120 | 5.000 | 5.000 | 0.62 | 0.697 |

CPU and GPU invocations can include overlapping test items; do not sum them as unique tests.
Public summaries omit raw logs, traces, paths, credentials, GPU UUIDs and provider authorization fields.

#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]
#![allow(non_snake_case)]
mod audio {
//! BEN94Z audio engine.
//!
//! Capture (CPAL) -> overlapped FFT (RustFFT) -> band analysis with automatic
//! gain, noise-floor tracking, attack/release envelopes, spectral-flux onset
//! detection (kick / snare / hi-hat), an approximate tempo estimate and
//! silence detection.
//!
//! All analysis runs inside the audio callback on state it owns exclusively, so
//! the render thread never blocks it. The finished `Levels` snapshot is
//! published through a mutex that is held only for one small copy; the render
//! thread reads it with `try_lock` and simply keeps the previous snapshot if
//! the lock happens to be busy.
#![allow(dead_code)]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

pub const FFT_SIZE: usize = 2048; // power of two (index masking below relies on it)
pub const HOP: usize = 512; // ~86 analysis frames per second at 44.1 kHz
pub const BANDS: usize = 64; // log-spaced display spectrum
const BINS: usize = FFT_SIZE / 2 + 1;
const ENV_LEN: usize = 640; // onset-envelope history, ~7 s
const WARMUP_HOPS: u32 = 12;
const TEMPO_EVERY_HOPS: u32 = 43; // ~0.5 s

/// Everything the visuals can react to. Cheap to copy (about 300 bytes).
///
/// Continuous values are 0..1 and already gain-normalised and smoothed.
/// Pulses (`kick`, `snare`, `hihat`, `beat`) jump up on a detected hit and
/// decay. Counters increase by one per hit, so a renderer running at 30 fps
/// can detect "a kick happened since my last frame" by comparing counters and
/// never misses one.
#[derive(Clone, Copy, Debug)]
pub struct Levels {
    pub volume: f32,
    pub sub: f32,       // ~20-60 Hz
    pub bass: f32,      // sub + 60-250 Hz
    pub mid: f32,       // 250 Hz - 3 kHz
    pub treble: f32,    // 3-16 kHz
    pub air: f32,       // 8-16 kHz shimmer
    pub vocal: f32,     // rough estimate: sustained 250 Hz - 3 kHz energy minus percussion
    pub kick: f32,
    pub snare: f32,
    pub hihat: f32,
    pub flux: f32,      // overall spectral change
    pub brightness: f32, // spectral centroid, 0 = dark, 1 = bright
    pub energy: f32,    // slow musical intensity (seconds)
    pub density: f32,   // how busy the rhythm is
    pub bpm: f32,       // approximate; trust only if tempo_conf is high
    pub tempo_conf: f32,
    pub beat_phase: f32, // 0..1 within a beat when tempo is locked
    pub beat: f32,      // pulse on beats (tempo grid if locked, otherwise kicks)
    pub kick_count: u32,
    pub snare_count: u32,
    pub hat_count: u32,
    pub beat_count: u32,
    pub silent: bool,
    pub spectrum: [f32; BANDS],
}

impl Levels {
    pub const ZERO: Levels = Levels {
        volume: 0.0, sub: 0.0, bass: 0.0, mid: 0.0, treble: 0.0, air: 0.0, vocal: 0.0,
        kick: 0.0, snare: 0.0, hihat: 0.0, flux: 0.0, brightness: 0.5, energy: 0.0,
        density: 0.0, bpm: 0.0, tempo_conf: 0.0, beat_phase: 0.0, beat: 0.0,
        kick_count: 0, snare_count: 0, hat_count: 0, beat_count: 0,
        silent: true, spectrum: [0.0; BANDS],
    };
}
impl Default for Levels {
    fn default() -> Self { Self::ZERO }
}

/// One-pole follower with separate attack and release time constants.
#[derive(Clone, Copy)]
struct Follow { up: f32, down: f32 }
impl Follow {
    fn new(attack: f32, release: f32, dt: f32) -> Self {
        Self { up: 1.0 - (-dt / attack).exp(), down: 1.0 - (-dt / release).exp() }
    }
    #[inline]
    fn step(&self, old: f32, new: f32) -> f32 {
        old + (new - old) * if new > old { self.up } else { self.down }
    }
}

/// Automatic gain + noise floor for one signal. The floor drops instantly to
/// the quietest recent value and creeps up slowly; the peak jumps up instantly
/// and decays slowly toward the floor. The output is the position between them.
#[derive(Clone, Copy)]
struct Agc { floor: f32, peak: f32, min_span: f32 }
impl Agc {
    fn new(min_span: f32) -> Self { Self { floor: 0.0, peak: min_span, min_span } }
    fn step(&mut self, e: f32, dt: f32) -> f32 {
        if e > self.peak { self.peak = e; } else { self.peak += (self.floor - self.peak) * (dt / 8.0); }
        if e < self.floor { self.floor = e; } else { self.floor += (e - self.floor) * (dt / 30.0); }
        let span = (self.peak - self.floor).max(self.min_span);
        ((e - self.floor) / span).clamp(0.0, 1.0)
    }
}

/// Spectral-flux onset detector with an adaptive threshold.
struct Onset { avg: f32, cooldown: f32, thr: f32, floor: f32, gap: f32, k: f32 }
impl Onset {
    fn new(thr: f32, floor: f32, gap: f32, dt: f32) -> Self {
        Self { avg: 0.0, cooldown: 0.0, thr, floor, gap, k: dt.min(1.0) }
    }
    /// Returns 0 for no hit, otherwise a strength in 0.15..=1.
    fn step(&mut self, flux: f32, dt: f32, armed: bool) -> f32 {
        self.cooldown -= dt;
        let thresh = self.avg * self.thr + self.floor;
        let mut hit = 0.0;
        if armed && flux > thresh && self.cooldown <= 0.0 {
            self.cooldown = self.gap;
            hit = ((flux - thresh) / (thresh + 1e-3)).clamp(0.15, 1.0);
        }
        self.avg += (flux - self.avg) * self.k;
        hit
    }
}

#[inline]
fn blend(a: f32, b: f32) -> f32 { 0.6 * a.max(b) + 0.2 * (a + b) }

fn mean_of(a: &[f32], lo: usize, hi: usize) -> f32 {
    a[lo..hi].iter().sum::<f32>() / (hi - lo).max(1) as f32
}

fn mean_flux(lm: &[f32], prev: &[f32], lo: usize, hi: usize) -> f32 {
    let mut s = 0.0;
    for k in lo..hi {
        let d = lm[k] - prev[k];
        if d > 0.0 { s += d; }
    }
    s / (hi - lo).max(1) as f32
}

struct Analyzer {
    fft: Arc<dyn Fft<f32>>,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    window: Vec<f32>,
    ring: Vec<f32>,
    widx: usize,
    count: usize,
    since_hop: usize,
    hops: u32,
    mags: Vec<f32>,
    lm: Vec<f32>,
    prev: Vec<f32>,
    dt: f32,
    binhz: f32,

    band_bins: [(usize, usize); 6],
    kick_range: (usize, usize),
    snare_range: (usize, usize),
    hat_range: (usize, usize),
    all_range: (usize, usize),
    centroid_range: (usize, usize),
    spec_edges: [(f32, f32); BANDS],
    spec_tilt: [f32; BANDS],

    agc: [Agc; 6],
    agc_vol: Agc,
    agc_flux: Agc,
    spec_peak: f32,
    spec_decay: f32,

    f_sub: Follow, f_bass: Follow, f_mid: Follow, f_treble: Follow, f_air: Follow,
    f_vol: Follow, f_spec: Follow, f_flux: Follow, f_slow: Follow, f_energy: Follow,
    decay_kick: f32, decay_snare: f32, decay_hat: f32, decay_beat: f32,

    kick_on: Onset, snare_on: Onset, hat_on: Onset,
    rate: f32,
    silence_timer: f32,

    env: Vec<f32>,
    env_lin: Vec<f32>,
    ac: Vec<f32>,
    env_w: usize,
    env_filled: usize,
    bpm: f32,
    cand: f32,
    cand_hits: u32,
    phase: f32,

    lv: Levels,
    sens: Arc<AtomicU32>,
    shared: Arc<Mutex<Levels>>,
}

impl Analyzer {
    fn new(sample_rate: f32, shared: Arc<Mutex<Levels>>, sens: Arc<AtomicU32>) -> Self {
        let sr = sample_rate.max(8000.0);
        let dt = HOP as f32 / sr;
        let binhz = sr / FFT_SIZE as f32;
        let nyq = sr * 0.5;
        let top = (nyq * 0.95).min(16000.0);
        let bin = |f: f32| ((f / binhz).round() as usize).clamp(1, BINS - 1);

        let edges = [20.0, 60.0, 250.0, 800.0, 3000.0, 8000.0_f32, 16000.0];
        let mut band_bins = [(1usize, 2usize); 6];
        for b in 0..6 {
            let lo = bin(edges[b].min(top * 0.8));
            let hi = bin(edges[b + 1].min(top)).max(lo + 1).min(BINS);
            band_bins[b] = (lo, hi);
        }
        let range = |a: f32, b: f32| {
            let lo = bin(a.min(top * 0.5));
            (lo, bin(b.min(top)).max(lo + 1).min(BINS))
        };

        let f0 = 30.0_f32;
        let f1 = top.min(16000.0);
        let mut spec_edges = [(0.0f32, 0.0f32); BANDS];
        let mut spec_tilt = [1.0f32; BANDS];
        for i in 0..BANDS {
            let a = f0 * (f1 / f0).powf(i as f32 / BANDS as f32);
            let b = f0 * (f1 / f0).powf((i + 1) as f32 / BANDS as f32);
            spec_edges[i] = (a / binhz, b / binhz);
            spec_tilt[i] = ((a * b).sqrt() / 250.0).powf(0.4).clamp(0.5, 4.0);
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let window = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (FFT_SIZE - 1) as f32).cos())
            .collect();

        Self {
            fft, scratch, window,
            buf: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            ring: vec![0.0; FFT_SIZE], widx: 0, count: 0, since_hop: 0, hops: 0,
            mags: vec![0.0; BINS], lm: vec![0.0; BINS], prev: vec![0.0; BINS],
            dt, binhz,
            band_bins,
            kick_range: range(35.0, 150.0),
            snare_range: range(250.0, 3500.0),
            hat_range: range(6000.0, 16000.0),
            all_range: range(60.0, 8000.0),
            centroid_range: range(100.0, 8000.0),
            spec_edges, spec_tilt,
            agc: [Agc::new(3e-4); 6],
            agc_vol: Agc::new(2e-3),
            agc_flux: Agc::new(0.02),
            spec_peak: 5e-4,
            spec_decay: (-dt / 6.0).exp(),
            f_sub: Follow::new(0.012, 0.20, dt),
            f_bass: Follow::new(0.015, 0.16, dt),
            f_mid: Follow::new(0.020, 0.12, dt),
            f_treble: Follow::new(0.010, 0.09, dt),
            f_air: Follow::new(0.008, 0.07, dt),
            f_vol: Follow::new(0.03, 0.25, dt),
            f_spec: Follow::new(0.02, 0.12, dt),
            f_flux: Follow::new(0.01, 0.10, dt),
            f_slow: Follow::new(0.6, 0.6, dt),
            f_energy: Follow::new(1.2, 3.5, dt),
            decay_kick: (-dt / 0.16).exp(),
            decay_snare: (-dt / 0.14).exp(),
            decay_hat: (-dt / 0.07).exp(),
            decay_beat: (-dt / 0.20).exp(),
            kick_on: Onset::new(1.8, 0.25, 0.12, dt),
            snare_on: Onset::new(1.7, 0.03, 0.12, dt),
            hat_on: Onset::new(1.8, 0.03, 0.05, dt),
            rate: 0.0, silence_timer: 0.0,
            env: vec![0.0; ENV_LEN], env_lin: vec![0.0; ENV_LEN], ac: vec![0.0; ENV_LEN + 2],
            env_w: 0, env_filled: 0,
            bpm: 0.0, cand: 0.0, cand_hits: 0, phase: 0.0,
            lv: Levels::ZERO, sens, shared,
        }
    }

    #[inline]
    fn push(&mut self, s: f32) {
        let s = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        self.ring[self.widx] = s;
        self.widx = (self.widx + 1) & (FFT_SIZE - 1);
        self.count = (self.count + 1).min(FFT_SIZE);
        self.since_hop += 1;
        if self.since_hop >= HOP && self.count >= FFT_SIZE {
            self.since_hop = 0;
            self.process();
        }
    }

    fn process(&mut self) {
        const MASK: usize = FFT_SIZE - 1;
        let dt = self.dt;
        let gain = (f32::from_bits(self.sens.load(Ordering::Relaxed)) * 0.5).clamp(0.05, 6.0);

        // Window the newest FFT_SIZE samples (widx is the oldest) and get RMS of the newest hop.
        for i in 0..FFT_SIZE {
            let v = self.ring[(self.widx + i) & MASK];
            self.buf[i] = Complex::new(v * self.window[i], 0.0);
        }
        let mut sq = 0.0;
        for j in 0..HOP {
            let v = self.ring[(self.widx + FFT_SIZE - HOP + j) & MASK];
            sq += v * v;
        }
        let rms = (sq / HOP as f32).sqrt();
        self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
        // A full-scale sine reads about 1.0 in its bin (Hann window).
        let scale = 4.0 / FFT_SIZE as f32;
        for k in 0..BINS { self.mags[k] = self.buf[k].norm() * scale; }

        // --- silence ---------------------------------------------------------
        if rms < 0.0010 { self.silence_timer += dt; } else { self.silence_timer = 0.0; }
        let silent = self.silence_timer > 0.8;
        self.hops = self.hops.saturating_add(1);
        let armed = self.hops > WARMUP_HOPS && !silent;

        // --- six analysis bands with automatic gain -----------------------------
        let mut v = [0f32; 6];
        for b in 0..6 {
            let (lo, hi) = self.band_bins[b];
            let mut s = 0.0;
            for k in lo..hi { s += self.mags[k]; }
            let n = self.agc[b].step(s / (hi - lo) as f32, dt);
            v[b] = if silent { 0.0 } else { (n * gain).min(1.0) };
        }
        let vol_n = self.agc_vol.step(rms, dt);
        let vol = if silent { 0.0 } else { (vol_n * gain).min(1.0) };

        // --- onset detection (spectral flux on log-compressed magnitudes) -----------
        for k in 0..BINS { self.lm[k] = (400.0 * self.mags[k]).ln_1p(); }
        let kf = mean_flux(&self.lm, &self.prev, self.kick_range.0, self.kick_range.1);
        let sf = mean_flux(&self.lm, &self.prev, self.snare_range.0, self.snare_range.1);
        let hf = mean_flux(&self.lm, &self.prev, self.hat_range.0, self.hat_range.1);
        let af = mean_flux(&self.lm, &self.prev, self.all_range.0, self.all_range.1);
        self.prev.copy_from_slice(&self.lm);

        // Cross-talk gates: a kick must dominate the low end over the mids (broadband noise
        // bursts are not kicks); a snare must clearly out-flux both the kick's bleed into the
        // mids and the hat band's rise (a hat transient is not a snare).
        let ek = mean_of(&self.mags, self.kick_range.0, self.kick_range.1);
        let es = mean_of(&self.mags, self.snare_range.0, self.snare_range.1);
        let kf = if ek > 2.0 * es { kf } else { 0.0 };
        let sf = if sf < 0.2 * kf || sf < 0.7 * hf { 0.0 } else { sf };
        let k_hit = self.kick_on.step(kf, dt, armed);
        let s_hit = self.snare_on.step(sf, dt, armed);
        let h_hit = self.hat_on.step(hf, dt, armed);

        let lv = &mut self.lv;
        lv.kick *= self.decay_kick;
        lv.snare *= self.decay_snare;
        lv.hihat *= self.decay_hat;
        lv.beat *= self.decay_beat;
        if k_hit > 0.0 { lv.kick_count = lv.kick_count.wrapping_add(1); lv.kick = lv.kick.max(0.55 + 0.45 * k_hit); }
        if s_hit > 0.0 { lv.snare_count = lv.snare_count.wrapping_add(1); lv.snare = lv.snare.max(0.55 + 0.45 * s_hit); }
        if h_hit > 0.0 { lv.hat_count = lv.hat_count.wrapping_add(1); lv.hihat = lv.hihat.max(0.5 + 0.5 * h_hit); }

        // --- continuous levels -----------------------------------------------------------
        lv.sub = self.f_sub.step(lv.sub, v[0]);
        lv.bass = self.f_bass.step(lv.bass, v[1].max(v[0] * 0.85).powf(1.2));
        let mid_t = blend(v[2], v[3]);
        lv.mid = self.f_mid.step(lv.mid, mid_t);
        lv.treble = self.f_treble.step(lv.treble, blend(v[4], v[5]));
        lv.air = self.f_air.step(lv.air, v[5]);
        lv.volume = self.f_vol.step(lv.volume, vol);
        let vocal_t = mid_t * (1.0 - 0.7 * lv.snare.max(lv.kick * 0.5));
        lv.vocal = self.f_mid.step(lv.vocal, vocal_t);
        let flux_n = if silent { 0.0 } else { self.agc_flux.step(af, dt) };
        lv.flux = self.f_flux.step(lv.flux, flux_n);

        // --- spectral centroid -> brightness ------------------------------------------------
        let (c0, c1) = self.centroid_range;
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for k in c0..c1 { num += k as f32 * self.mags[k]; den += self.mags[k]; }
        if den > 1e-4 && !silent {
            let centroid = num / den * self.binhz;
            let b = ((centroid / 200.0).max(1.0).log2() / (4000.0f32 / 200.0).log2()).clamp(0.0, 1.0);
            lv.brightness = self.f_slow.step(lv.brightness, b);
        }

        // --- log-spaced display spectrum -------------------------------------------------------
        let mut raw = [0f32; BANDS];
        let mut peak_now = 0f32;
        for i in 0..BANDS {
            let (lo, hi) = self.spec_edges[i];
            let val = if hi - lo < 1.0 {
                let c = 0.5 * (lo + hi);
                let i0 = c.floor() as usize;
                let fr = c - i0 as f32;
                let i1 = (i0 + 1).min(BINS - 1);
                self.mags[i0] * (1.0 - fr) + self.mags[i1] * fr
            } else {
                let a = lo.floor() as usize;
                let b = (hi.ceil() as usize).min(BINS);
                let mut s = 0.0;
                for k in a..b { s += self.mags[k]; }
                s / (b - a).max(1) as f32
            };
            let val = val * self.spec_tilt[i];
            raw[i] = val;
            peak_now = peak_now.max(val);
        }
        self.spec_peak = (self.spec_peak * self.spec_decay).max(peak_now).max(5e-4);
        for i in 0..BANDS {
            let t = if silent { 0.0 } else { ((raw[i] / self.spec_peak).powf(0.8) * gain).min(1.0) };
            lv.spectrum[i] = self.f_spec.step(lv.spectrum[i], t);
        }

        // --- rhythm density and slow energy ----------------------------------------------------------
        let hits = (k_hit > 0.0) as u32 as f32 + (s_hit > 0.0) as u32 as f32 + 0.5 * (h_hit > 0.0) as u32 as f32;
        self.rate += (hits / dt - self.rate) * (dt / 2.0);
        lv.density = (self.rate / 5.0).clamp(0.0, 1.0);
        let e_t = 0.45 * lv.volume + 0.35 * (lv.bass + lv.mid + lv.treble) / 3.0 + 0.20 * lv.density;
        lv.energy = self.f_energy.step(lv.energy, if silent { 0.0 } else { e_t });
        lv.silent = silent;

        // --- tempo estimate and beat clock ------------------------------------------------------------
        self.env[self.env_w] = kf + 0.7 * sf + 0.3 * hf;
        self.env_w = (self.env_w + 1) % ENV_LEN;
        self.env_filled = (self.env_filled + 1).min(ENV_LEN);
        if self.hops % TEMPO_EVERY_HOPS == 0 && !silent { self.estimate_tempo(); }
        if silent { self.lv.tempo_conf *= 0.95; }

        let locked = self.lv.tempo_conf > 0.2 && self.bpm > 0.0;
        let mut beat_hit = false;
        if locked {
            self.phase += dt * self.bpm / 60.0;
            if k_hit > 0.4 {
                let mut err = self.phase;
                if err > 0.5 { err -= 1.0; }
                if err.abs() < 0.22 { self.phase -= err * 0.2; }
            }
            if self.phase >= 1.0 { self.phase -= 1.0; beat_hit = true; }
            if self.phase < 0.0 { self.phase += 1.0; }
            self.lv.beat_phase = self.phase;
        } else {
            self.lv.beat_phase = 0.0;
            beat_hit = k_hit > 0.0;
        }
        if beat_hit {
            self.lv.beat_count = self.lv.beat_count.wrapping_add(1);
            self.lv.beat = 1.0;
        }
        self.lv.bpm = self.bpm;

        if let Ok(mut g) = self.shared.lock() { *g = self.lv; }
    }

    /// Autocorrelation of the onset envelope over 60-190 BPM with a mild prior
    /// toward 120 BPM. Approximate: halving/doubling errors are possible.
    fn estimate_tempo(&mut self) {
        let len = self.env_filled;
        if len < 256 { return; }
        let start = (self.env_w + ENV_LEN - len) % ENV_LEN;
        let mut mean = 0.0;
        for i in 0..len {
            let v = self.env[(start + i) % ENV_LEN];
            self.env_lin[i] = v;
            mean += v;
        }
        mean /= len as f32;
        let mut var0 = 0.0;
        for i in 0..len {
            self.env_lin[i] -= mean;
            var0 += self.env_lin[i] * self.env_lin[i];
        }
        if var0 < 1e-5 { self.lv.tempo_conf *= 0.8; return; }

        let lag_min = ((60.0 / 190.0 / self.dt).floor() as usize).max(2);
        let lag_max = ((60.0 / 60.0 / self.dt).ceil() as usize).min(len / 2);
        if lag_max <= lag_min + 2 { return; }
        let mut best = 0.0;
        let mut best_lag = 0usize;
        for lag in lag_min..=lag_max {
            let mut s = 0.0;
            for i in 0..len - lag { s += self.env_lin[i] * self.env_lin[i + lag]; }
            let norm = s / (len - lag) as f32 / (var0 / len as f32);
            self.ac[lag] = norm;
            let bpm = 60.0 / (lag as f32 * self.dt);
            let prior = (-0.5 * ((bpm / 120.0).ln() / 0.6).powi(2)).exp();
            let w = norm * prior;
            if w > best { best = w; best_lag = lag; }
        }
        if best_lag == 0 { return; }
        let conf = self.ac[best_lag].clamp(0.0, 1.0);
        let mut off = 0.0;
        if best_lag > lag_min && best_lag < lag_max {
            let (a, b, c) = (self.ac[best_lag - 1], self.ac[best_lag], self.ac[best_lag + 1]);
            let d = a - 2.0 * b + c;
            if d.abs() > 1e-6 { off = (0.5 * (a - c) / d).clamp(-1.0, 1.0); }
        }
        let new_bpm = 60.0 / ((best_lag as f32 + off) * self.dt);
        self.lv.tempo_conf += (conf - self.lv.tempo_conf) * 0.5;
        if conf > 0.12 {
            if self.bpm <= 0.0 {
                self.bpm = new_bpm;
            } else if (new_bpm - self.bpm).abs() / self.bpm < 0.06 {
                self.bpm += (new_bpm - self.bpm) * 0.3;
                self.cand_hits = 0;
            } else {
                // A different tempo must be seen three times in a row before we switch.
                if (new_bpm - self.cand).abs() / self.cand.max(1.0) < 0.05 {
                    self.cand_hits += 1;
                } else {
                    self.cand = new_bpm;
                    self.cand_hits = 1;
                }
                if self.cand_hits >= 3 { self.bpm = self.cand; self.cand_hits = 0; }
            }
        }
    }
}

/// Handle owned by the render thread.
pub struct AudioInput {
    shared: Arc<Mutex<Levels>>,
    sens: Arc<AtomicU32>,
    last: Levels,
    stream: Option<cpal::Stream>,
    demo: bool,
    pub device_name: String,
}

impl AudioInput {
    /// Never fails: with no usable input device the visuals simply see silence.
    pub fn start() -> Self {
        let shared = Arc::new(Mutex::new(Levels::ZERO));
        let sens = Arc::new(AtomicU32::new(2.0f32.to_bits()));
        let mut device_name = String::from("none");
        if std::env::var("BEN94Z_DEMO").map(|v| v != "0").unwrap_or(false) {
            start_demo(shared.clone(), sens.clone());
            return Self { shared, sens, last: Levels::ZERO, stream: None, demo: true, device_name: "demo track".into() };
        }
        let stream = match open_stream(shared.clone(), sens.clone(), &mut device_name) {
            Ok(s) => Some(s),
            Err(e) => { eprintln!("BEN94Z audio: {e} (running without audio input)"); None }
        };
        Self { shared, sens, last: Levels::ZERO, stream, demo: false, device_name }
    }

    /// Latest snapshot. Never blocks; reuses the previous one if the lock is busy.
    pub fn poll(&mut self) -> &Levels {
        if let Ok(l) = self.shared.try_lock() { self.last = *l; }
        &self.last
    }

    pub fn is_live(&self) -> bool { self.stream.is_some() || self.demo }
    pub fn is_demo(&self) -> bool { self.demo }
    pub fn sensitivity(&self) -> f32 { f32::from_bits(self.sens.load(Ordering::Relaxed)) }
    pub fn adjust_sensitivity(&self, delta: f32) {
        let v = (self.sensitivity() + delta).clamp(0.1, 10.0);
        self.sens.store(v.to_bits(), Ordering::Relaxed);
    }
}

/// Synthetic 124 BPM track (`BEN94Z_DEMO=1`): lets you preview every scene without any audio
/// input. It runs through exactly the same analyzer as real audio. The arrangement cycles
/// through pad-only, groove, full kit with a lead, and a breakdown, so the scene director has
/// something to react to.
fn demo_sample(t: f32, noise: &mut u32, prev: &mut f32) -> f32 {
    use std::f32::consts::TAU;
    let beat = 60.0 / 124.0;
    let bt = t % beat;
    let step = (t / beat * 4.0) as usize; // 16th notes
    let bar = (t / (beat * 4.0)) as usize;
    let section = (bar / 4) % 4;
    *noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
    let nz = (*noise >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    let hp = nz - *prev;
    *prev = nz;
    let mut s = 0.0;
    // pad chord, always present but quiet
    let lfo = 0.6 + 0.4 * (t * 0.3).sin();
    for f in [220.0f32, 277.2, 329.6] { s += (TAU * f * t).sin() * 0.05 * lfo; }
    if section == 1 || section == 2 {
        s += (TAU * (50.0 + 80.0 * (-bt * 40.0).exp()) * bt).sin() * (-bt * 14.0).exp() * 0.8; // kick
        s += (TAU * 55.0 * t).sin() * 0.18 * (0.4 + 0.6 * (bt / beat)); // bass note swelling between kicks
        let ht = t % (beat / 2.0);
        s += hp * (-ht * 90.0).exp() * 0.22; // hats
    }
    if section == 2 {
        if (t / beat) as usize % 4 == 1 || (t / beat) as usize % 4 == 3 {
            s += nz * (-bt * 22.0).exp() * 0.30 + (TAU * 190.0 * bt).sin() * (-bt * 25.0).exp() * 0.25; // snare
        }
        let notes = [440.0f32, 523.3, 659.3, 587.3, 523.3, 784.0, 659.3, 523.3];
        let nt = (t / beat * 4.0).fract();
        s += (TAU * notes[step % 8] * t).sin() * 0.10 * (-nt * 3.0).exp(); // arpeggio lead
    }
    if section == 0 {
        let ht = t % beat;
        s += hp * (-ht * 70.0).exp() * 0.08;
    }
    s
}

fn start_demo(shared: Arc<Mutex<Levels>>, sens: Arc<AtomicU32>) {
    let mut an = Analyzer::new(44100.0, shared, sens);
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let (mut done, mut noise, mut prev) = (0u64, 0x1234_5678u32, 0.0f32);
        loop {
            let target = (start.elapsed().as_secs_f64() * 44100.0) as u64;
            while done < target {
                an.push(demo_sample(done as f32 / 44100.0, &mut noise, &mut prev));
                done += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
}

fn pick_device(host: &cpal::Host) -> Option<cpal::Device> {
    // Optional: BEN94Z_AUDIO_DEVICE=monitor picks the first input whose name contains "monitor".
    if let Ok(want) = std::env::var("BEN94Z_AUDIO_DEVICE") {
        let want = want.to_lowercase();
        if let Ok(devs) = host.input_devices() {
            for d in devs {
                if d.name().map(|n| n.to_lowercase().contains(&want)).unwrap_or(false) { return Some(d); }
            }
        }
        eprintln!("BEN94Z audio: no input device matching '{want}', using the default");
    }
    host.default_input_device()
}

/// Picks the capture device. On Windows the default is WASAPI loopback of the default output
/// device (cpal opens a render device as a loopback capture), so the visuals follow whatever the
/// PC is playing without needing "Stereo Mix". Set BEN94Z_MIC=1 or BEN94Z_AUDIO_DEVICE=<name> to
/// use a microphone or a specific input instead. Everywhere else the default input is used.
fn choose_device(host: &cpal::Host) -> Result<(cpal::Device, cpal::SupportedStreamConfig, bool), String> {
    #[cfg(target_os = "windows")]
    {
        if std::env::var("BEN94Z_AUDIO_DEVICE").is_err() && std::env::var("BEN94Z_MIC").is_err() {
            if let Some(d) = host.default_output_device() {
                if let Ok(c) = d.default_output_config() { return Ok((d, c, true)); }
            }
        }
    }
    let device = pick_device(host).ok_or("no audio input device")?;
    let config = device.default_input_config().map_err(|e| format!("no input config: {e}"))?;
    Ok((device, config, false))
}

fn open_stream(shared: Arc<Mutex<Levels>>, sens: Arc<AtomicU32>, name_out: &mut String) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let (device, config, loopback) = choose_device(&host)?;
    let channels = (config.channels() as usize).max(1);
    let sr = config.sample_rate().0 as f32;
    *name_out = device.name().unwrap_or_default();
    println!("Audio {}: {} ({} Hz, {} ch)", if loopback { "loopback (what this PC is playing)" } else { "input" }, name_out, sr as u32, channels);
    let cfg: cpal::StreamConfig = config.clone().into();
    let mut an = Analyzer::new(sr, shared, sens);
    let err = |e| eprintln!("Audio stream error: {e}");
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(&cfg, move |data: &[f32], _: &cpal::InputCallbackInfo| {
            for c in data.chunks(channels) { an.push(c.iter().sum::<f32>() / c.len() as f32); }
        }, err, None),
        cpal::SampleFormat::I16 => device.build_input_stream(&cfg, move |data: &[i16], _: &cpal::InputCallbackInfo| {
            for c in data.chunks(channels) { an.push(c.iter().map(|&v| v as f32 / 32768.0).sum::<f32>() / c.len() as f32); }
        }, err, None),
        cpal::SampleFormat::U16 => device.build_input_stream(&cfg, move |data: &[u16], _: &cpal::InputCallbackInfo| {
            for c in data.chunks(channels) { an.push(c.iter().map(|&v| v as f32 / 32768.0 - 1.0).sum::<f32>() / c.len() as f32); }
        }, err, None),
        other => return Err(format!("unsupported sample format {other:?}")),
    }
    .map_err(|e| format!("cannot open input stream: {e}"))?;
    stream.play().map_err(|e| format!("cannot start input stream: {e}"))?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    const SR: f32 = 44100.0;

    fn make() -> (Analyzer, Arc<Mutex<Levels>>) {
        let shared = Arc::new(Mutex::new(Levels::ZERO));
        let sens = Arc::new(AtomicU32::new(2.0f32.to_bits()));
        (Analyzer::new(SR, shared.clone(), sens), shared)
    }
    struct Noise(u32);
    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        }
    }
    /// Synthetic drum loop: kick on every beat, snare on 2 and 4, optional 8th-note hats.
    fn drums(an: &mut Analyzer, bpm: f32, secs: f32, amp: f32, kick: bool, snare: bool, hats: bool) {
        let beat = 60.0 / bpm;
        let mut noise = Noise(12345);
        let mut prev_n = 0.0f32;
        for n in 0..(secs * SR) as usize {
            let t = n as f32 / SR;
            let bt = t % beat;
            let bar_beat = ((t / beat) as usize) % 4;
            let mut s = 0.0;
            if kick { s += (std::f32::consts::TAU * (50.0 + 80.0 * (-bt * 40.0).exp()) * bt).sin() * (-bt * 14.0).exp() * 0.8; }
            let nz = noise.next();
            let hp = nz - prev_n; // crude high-pass
            prev_n = nz;
            if snare && (bar_beat == 1 || bar_beat == 3) { s += nz * (-bt * 22.0).exp() * 0.35; s += (std::f32::consts::TAU * 190.0 * bt).sin() * (-bt * 25.0).exp() * 0.3; }
            if hats { let ht = t % (beat / 2.0); s += hp * (-ht * 90.0).exp() * 0.25; }
            an.push(s * amp);
        }
    }

    #[test]
    fn kick_loop_detects_kicks_and_tempo() {
        let (mut an, shared) = make();
        drums(&mut an, 120.0, 20.0, 1.0, true, false, false);
        let lv = *shared.lock().unwrap();
        println!("kicks={} bpm={:.1} conf={:.2} bass={:.2}", lv.kick_count, lv.bpm, lv.tempo_conf, lv.bass);
        assert!((34..=46).contains(&lv.kick_count), "kick_count {}", lv.kick_count);
        assert!(lv.tempo_conf > 0.2, "conf {}", lv.tempo_conf);
        assert!((lv.bpm - 120.0).abs() < 4.0, "bpm {}", lv.bpm);
        assert!(!lv.silent);
    }

    #[test]
    fn full_kit_separates_instruments() {
        let (mut an, shared) = make();
        drums(&mut an, 128.0, 16.0, 0.8, true, true, true);
        let lv = *shared.lock().unwrap();
        println!("kick={} snare={} hat={} bpm={:.1} conf={:.2}", lv.kick_count, lv.snare_count, lv.hat_count, lv.bpm, lv.tempo_conf);
        assert!((25..=40).contains(&lv.kick_count), "kicks {}", lv.kick_count);
        assert!((12..=24).contains(&lv.snare_count), "snares {}", lv.snare_count);
        assert!((40..=90).contains(&lv.hat_count), "hats {}", lv.hat_count);
        assert!((lv.bpm - 128.0).abs() < 5.0, "bpm {}", lv.bpm);
    }

    #[test]
    fn hats_alone_do_not_trigger_kicks() {
        let (mut an, shared) = make();
        drums(&mut an, 120.0, 10.0, 1.0, false, false, true);
        let lv = *shared.lock().unwrap();
        println!("kick={} hat={}", lv.kick_count, lv.hat_count);
        assert_eq!(lv.kick_count, 0);
        assert!(lv.hat_count >= 10);
    }

    #[test]
    fn detection_is_level_independent() {
        let (mut a1, s1) = make();
        let (mut a2, s2) = make();
        drums(&mut a1, 120.0, 16.0, 0.9, true, true, false);
        drums(&mut a2, 120.0, 16.0, 0.06, true, true, false);
        let (l1, l2) = (*s1.lock().unwrap(), *s2.lock().unwrap());
        println!("loud kicks={} quiet kicks={}", l1.kick_count, l2.kick_count);
        assert!((l1.kick_count as i32 - l2.kick_count as i32).abs() <= 3);
        assert!(l2.kick_count >= 24);
    }

    #[test]
    fn silence_is_detected_and_quiet() {
        let (mut an, shared) = make();
        for _ in 0..(5.0 * SR) as usize { an.push(0.0); }
        let lv = *shared.lock().unwrap();
        assert!(lv.silent);
        assert_eq!((lv.kick_count, lv.snare_count, lv.hat_count), (0, 0, 0));
        assert!(lv.bass < 0.01 && lv.mid < 0.01 && lv.treble < 0.01 && lv.energy < 0.01);
        assert!(lv.spectrum.iter().all(|&s| s < 0.01));
    }

    #[test]
    fn sine_lands_in_the_right_band() {
        let (mut an, shared) = make();
        for n in 0..(4.0 * SR) as usize { an.push((std::f32::consts::TAU * 1000.0 * n as f32 / SR).sin() * 0.5); }
        let lv = *shared.lock().unwrap();
        println!("bass={:.2} mid={:.2} treble={:.2}", lv.bass, lv.mid, lv.treble);
        assert!(lv.mid > 0.5 && lv.bass < 0.1 && lv.treble < 0.2);
        let (imax, _) = lv.spectrum.iter().enumerate().fold((0, 0.0), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        // 1 kHz on a 30 Hz-16 kHz log axis of 64 bands is band ~ 64*log(1000/30)/log(16000/30) = 36
        assert!((33..=39).contains(&imax), "peak band {imax}");
        assert!(!lv.silent);
    }
}
}
mod details {
//! Extra detail layers drawn on top of the existing scenes.
//!
//! Each function adds new geometry to one theme, scales down to about a third in ECO,
//! uses stable hashed placement (no per-frame randomness, no allocation) and ties motion
//! to the music: bass -> large slow motion, mid -> lights, treble -> small sparkle,
//! kick/snare timing -> shockwaves and arcs.
#![allow(dead_code)]

use crate::audio::BANDS;
use crate::fx::*;
use macroquad::prelude::*;
use std::f32::consts::TAU;

// ---------------------------------------------------------------- RAINBOW (theme 3)
pub fn rainbow(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    let core = vec3(0., 6.0, 0.);
    let n = cnt(eco, 14);
    for i in 0..n {
        let u = i as u32;
        let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
        let a = t * (0.07 + 0.02 * (i % 3) as f32) * dir + i as f32 * 0.45;
        let r = 13.0 + (i % 4) as f32 * 3.0;
        let p = vec3(a.cos() * r, 7.0 + (i % 5) as f32 * 1.8 + (t * 0.6 + i as f32).sin() * 0.8, a.sin() * r);
        let hue = t * 0.04 + i as f32 * 0.071;
        octa(p, 0.9 + f.mid * 0.5, t * 0.5 + i as f32, hsv(hue, 0.7, 0.95, 1.0));
        if i % 2 == 0 {
            let c = hsv(hue, 0.6, 0.5 + 0.35 * f.kick, 1.0);
            draw_line_3d(core, p, c);
            if !eco { draw_line_3d(core + vec3(0.1, 0.1, 0.), p + vec3(0.1, 0.1, 0.), alpha(c, 0.5)); }
        }
        let _ = u;
    }
    // Hexagonal shockwaves on the floor: kick = wide, snare = tight.
    for (since, speed, life, sides, off) in [(f.since_kick, 20.0f32, 1.5f32, 6usize, 0.0f32), (f.since_snare, 28.0, 1.0, 6, 0.5)] {
        if since < life {
            let a = 1.0 - since / life;
            let c = hsv(t * 0.1 + off, 0.6, a, 1.0);
            let r = since * speed;
            for k in 0..sides {
                let a0 = TAU * k as f32 / sides as f32 + t * 0.2;
                let a1 = TAU * (k + 1) as f32 / sides as f32 + t * 0.2;
                draw_line_3d(vec3(a0.cos() * r, 0.05, a0.sin() * r), vec3(a1.cos() * r, 0.05, a1.sin() * r), c);
            }
        }
    }
    for i in 0..cnt(eco, 28) {
        let u = i as u32 + 77;
        let tw = (0.5 + 0.5 * (t * (1.0 + h01(u, 1) * 2.0) + h01(u, 2) * TAU).sin()).powi(3);
        let p = vec3((h01(u, 3) - 0.5) * 50.0, 1.0 + h01(u, 4) * 14.0, (h01(u, 5) - 0.5) * 50.0);
        spark(p, 0.1 + 0.3 * tw * (0.5 + f.treble), hsv(h01(u, 6) + t * 0.05, 0.4, 0.4 + 0.6 * tw, 1.0));
    }
}

// ---------------------------------------------------------------- IRON MAN (theme 4)
fn arm(base: Vec3, side: f32, f: &Frame) {
    let t = f.t;
    let yaw = (t * 0.3 + side * 2.0).sin() * 0.9 + if side < 0. { 0.4 } else { -0.4 };
    let d = vec3(yaw.cos(), 0., yaw.sin());
    let a1 = 1.0 + 0.35 * (t * 0.55 + side).sin();
    let a2 = a1 - 1.5 + 0.45 * (t * 0.8 + 1.0 + side).sin();
    let a3 = a2 - 0.6 + 0.4 * (t * 1.3 + side).sin();
    let seg = |from: Vec3, ang: f32, len: f32| from + d * (ang.cos() * len) + vec3(0., ang.sin() * len, 0.);
    let p1 = seg(base, a1, 7.0);
    let p2 = seg(p1, a2, 6.0);
    let p3 = seg(p2, a3, 3.0);
    let steel = Color::new(0.45, 0.55, 0.65, 1.0);
    let cyan = Color::new(0.18, 0.78, 1.0, 1.0);
    for off in [vec3(0., 0., 0.), vec3(0.18, 0., 0.18), vec3(-0.18, 0., -0.18)] {
        draw_line_3d(base + off, p1 + off, steel);
        draw_line_3d(p1 + off, p2 + off, steel);
        draw_line_3d(p2 + off, p3 + off, steel);
    }
    for p in [base, p1, p2] { spark(p, 0.55, cyan); }
    ring_h(base, 1.4, 12, 0., steel);
    draw_cube_wires(base - vec3(0., 0.5, 0.), vec3(2.2, 1.0, 2.2), steel);
    // Gripper prongs open a little on every snare.
    let open = 0.3 + 0.5 * f.snare;
    let perp = vec3(-d.z, 0., d.x);
    let tip_dir = (p3 - p2).normalize_or_zero();
    for s in [-1.0f32, 1.0] {
        draw_line_3d(p3, p3 + perp * (s * open) + tip_dir * 1.1, cyan);
    }
    if f.snare > 0.35 && !f.eco {
        let seed = (t * 30.0).floor() + side;
        bolt(p3 + tip_dir * 1.1, p3 + tip_dir * 1.1 + vec3(0., -1.5, 0.), seed, 0.5, Color::new(0.7, 0.95, 1.0, 1.0));
    }
}
pub fn iron(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    let steel = Color::new(0.45, 0.55, 0.65, 1.0);
    let cyan = Color::new(0.18, 0.78, 1.0, 1.0);
    arm(vec3(-14.0, -3.0, 10.0), -1.0, f);
    if !eco { arm(vec3(14.0, -3.0, 10.0), 1.0, f); }
    // Pistons with bass-assisted strokes.
    for i in 0..cnt(eco, 4) {
        let x = -18.0 + i as f32 * 12.0;
        let c = vec3(x, 0.0, -18.0);
        draw_cube_wires(c, vec3(1.6, 5.0, 1.6), steel);
        let stroke = 2.5 + 2.2 * (0.5 + 0.5 * (t * 1.7 + i as f32 * 1.3).sin()) + f.bass * 1.2;
        draw_line_3d(c + vec3(0., 2.5, 0.), c + vec3(0., 2.5 + stroke, 0.), cyan);
        draw_cube_wires(c + vec3(0., 2.5 + stroke, 0.), vec3(1.1, 0.25, 1.1), cyan);
    }
    // Holographic display: rotating arcs, spectrum bars and a sweeping scan ring.
    let hc = vec3(0., 4.5, -8.0);
    for k in 0..3 {
        let rot = t * [0.5f32, -0.7, 0.9][k];
        arc_h(hc + vec3(0., 1.0 + k as f32 * 0.5, 0.), 3.2 + k as f32 * 0.6, rot, rot + TAU * 0.75, if eco { 12 } else { 24 }, scale(cyan, 0.8));
    }
    for i in 0..cnt(eco, 16) {
        let a = TAU * i as f32 / cnt(eco, 16) as f32 + t * 0.15;
        let s = f.spectrum[(i * 64 / cnt(eco, 16)).min(BANDS - 1)];
        let p = hc + vec3(a.cos() * 2.4, 0., a.sin() * 2.4);
        draw_line_3d(p, p + vec3(0., 0.2 + s * 2.5, 0.), Color::new(0.3 + s * 0.5, 0.85, 1.0, 1.0));
    }
    ring_h(hc + vec3(0., (t * 0.5).fract() * 3.5, 0.), 2.6, 20, 0., alpha(cyan, 0.5));
    // Beacons: steady glow, a little brighter on each snare (limited range, never a strobe).
    for i in 0..cnt(eco, 6) {
        let x = if i % 2 == 0 { -19.0 } else { 19.0 };
        let z = [-19.0f32, 0.0, 19.0][i / 2 % 3];
        let p = vec3(x, 12.5, z);
        let k = 0.35 + 0.5 * f.snare;
        let a = t * 3.0 + i as f32;
        spark(p, 0.5, Color::new(1.0 * k, 0.12 * k, 0.08 * k, 1.0));
        draw_line_3d(p, p + vec3(a.cos() * 3.0, -0.8, a.sin() * 3.0), Color::new(0.8 * k, 0.1 * k, 0.05 * k, 1.0));
    }
}

// ---------------------------------------------------------------- SNOW (theme 5)
pub fn snow(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    // Aurora curtains, intensity follows the mids.
    for r in 0..cnt(eco, 3) {
        let hue = 0.38 + 0.12 * r as f32 + 0.06 * (t * 0.05 + r as f32).sin();
        let k = 0.45 + 0.4 * f.mid;
        let mut last = Vec3::ZERO;
        for i in 0..=30 {
            let x = -60.0 + i as f32 * 4.0;
            let y = 26.0 + r as f32 * 3.0 + (x * 0.1 + t * 0.2 + r as f32 * 1.7).sin() * 4.0 + (x * 0.25 - t * 0.3).sin() * 1.5;
            let p = vec3(x, y, -48.0 - r as f32 * 5.0);
            if i > 0 {
                draw_line_3d(last, p, hsv(hue, 0.6, k, 1.0));
            }
            if !eco || i % 2 == 0 {
                draw_line_3d(p, p - vec3(0., 7.0 + (x * 0.3 + t).sin() * 2.0, 0.), hsv(hue + 0.05, 0.7, k * 0.35, 1.0));
            }
            last = p;
        }
    }
    // Three layers of wind-driven snowfall (far / mid / near).
    let wind = (t * 0.2).sin() * 4.0 + f.mid * 3.0;
    for layer in 0..3u32 {
        let n = cnt(eco, 36);
        let speed = 2.0 + 1.2 * layer as f32;
        for i in 0..n {
            let u = i as u32 + layer * 100;
            let ph = (t * speed + h01(u, 1) * 28.0) % 28.0;
            let y = 28.0 - ph;
            let x = (h01(u, 2) - 0.5) * (60.0 + 20.0 * layer as f32) + (t * 0.4 + h01(u, 3) * TAU).sin() * 1.5 * (layer + 1) as f32 + wind * (1.0 - y / 28.0);
            let z = (h01(u, 4) - 0.5) * 70.0;
            let p = vec3(x, y, z);
            draw_line_3d(p, p + vec3(0.05 * wind, -0.14 - 0.06 * layer as f32, 0.), Color::new(0.85, 0.92, 1.0, 0.75));
        }
    }
    // Glints on the frozen lake, brighter with treble.
    for i in 0..cnt(eco, 24) {
        let u = i as u32 + 300;
        let tw = (0.5 + 0.5 * (t * 1.7 + h01(u, 1) * TAU).sin()).powi(4) * (0.4 + f.treble);
        spark(vec3((h01(u, 2) - 0.5) * 22.0, -0.6, (h01(u, 3) - 0.5) * 60.0), 0.15 + 0.3 * tw, Color::new(0.7 + 0.3 * tw, 0.9, 1.0, 1.0));
    }
}

// ---------------------------------------------------------------- MATRIX (theme 6)
pub fn matrix(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    let neon = Color::new(0.08, 0.95, 0.35, 1.0);
    // Packets racing along the floor grid; speed follows energy.
    let sp = 8.0 + 10.0 * f.energy;
    for i in 0..cnt(eco, 16) {
        let u = i as u32;
        let x = -24.0 + 6.0 * (i % 9) as f32;
        let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
        let z = ((t * sp * (0.6 + h01(u, 1) * 0.8) + h01(u, 2) * 60.0) % 60.0 - 30.0) * dir;
        draw_line_3d(vec3(x, -0.9, z), vec3(x, -0.9, z - dir * 3.0), Color::new(0.6, 1.0, 0.7, 1.0));
        draw_line_3d(vec3(x, -0.9, z - dir * 3.0), vec3(x, -0.9, z - dir * 7.0), scale(neon, 0.45));
    }
    // Floor shockwave on each kick: a square ring of light.
    if f.since_kick < 1.2 {
        let a = 1.0 - f.since_kick / 1.2;
        let r = f.since_kick * 36.0;
        let c = Color::new(0.2 * a, 1.0 * a, 0.5 * a, 1.0);
        let y = -0.9;
        let q = [vec3(-r, y, -r), vec3(r, y, -r), vec3(r, y, r), vec3(-r, y, r), vec3(-r, y, -r)];
        poly(&q, c);
    }
    // Network nodes pulse on hi-hats and are linked by travelling dashes.
    let nodes = [-18.0f32, 0.0, 18.0];
    for (ix, &x) in nodes.iter().enumerate() {
        for (iz, &z) in nodes.iter().enumerate() {
            let p = vec3(x, -0.4, z);
            spark(p, 0.45 + 0.7 * f.hat, scale(neon, 0.8));
            if !eco || iz == 0 {
                for (jx, jz) in [(ix + 1, iz), (ix, iz + 1)] {
                    if jx < 3 && jz < 3 {
                        let q = vec3(nodes[jx], -0.4, nodes[jz]);
                        draw_line_3d(p, q, scale(neon, 0.25));
                        let k = (t * 0.6 + (ix * 3 + iz) as f32 * 0.37).fract();
                        let a = p.lerp(q, k);
                        draw_line_3d(a, p.lerp(q, (k + 0.08).min(1.0)), Color::new(0.7, 1.0, 0.8, 1.0));
                    }
                }
            }
        }
    }
    // Holographic spectrum panel on the far wall.
    let c = vec3(0.0, 9.0, -27.0);
    let q = [c + vec3(-10., -4., 0.), c + vec3(10., -4., 0.), c + vec3(10., 4., 0.), c + vec3(-10., 4., 0.), c + vec3(-10., -4., 0.)];
    poly(&q, scale(neon, 0.6));
    let bars = cnt(eco, 24);
    for i in 0..bars {
        let s = f.spectrum[(i * 64 / bars).min(BANDS - 1)];
        let x = -9.4 + i as f32 * (18.8 / bars as f32);
        draw_line_3d(c + vec3(x, -3.6, 0.), c + vec3(x, -3.6 + 0.3 + s * 6.5, 0.), Color::new(0.1 + s * 0.4, 0.95, 0.5 + s * 0.4, 1.0));
    }
    let sy = -4.0 + (t * 0.4).fract() * 8.0;
    draw_line_3d(c + vec3(-10., sy, 0.), c + vec3(10., sy, 0.), alpha(neon, 0.35));
}

// ---------------------------------------------------------------- FIRE (theme 7)
pub fn fire(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    let wind = (t * 0.3).sin() * 1.5 + 0.5;
    // Rising embers.
    for i in 0..cnt(eco, 40) {
        let u = i as u32;
        let age = (t * (0.25 + f.treble * 0.2) + h01(u, 1)).fract();
        let a = h01(u, 2) * TAU;
        let p = vec3(a.cos() * h01(u, 3) * 1.4 + (t * 0.5 + h01(u, 4) * 6.0).sin() * age * 2.5 + wind * age * 2.0, 0.9 + age * 10.0, a.sin() * h01(u, 3) * 1.4);
        spark(p, 0.07 * (1.0 - age * 0.6), mixc(Color::new(1.0, 0.65, 0.12, 1.), Color::new(0.55, 0.08, 0.02, 1.), age));
    }
    // Spark burst and a low shockwave on each kick.
    if f.since_kick < 0.9 {
        let a = 1.0 - f.since_kick / 0.9;
        for i in 0..cnt(eco, 14) {
            let u = i as u32 + 90;
            let ang = h01(u, 1) * TAU;
            let sp = 3.0 + 5.0 * h01(u, 2);
            let s = f.since_kick;
            spark(vec3(ang.cos() * sp * s, 1.0 + 6.0 * s - 7.0 * s * s, ang.sin() * sp * s), 0.12 * a, Color::new(1.0, 0.7 * a, 0.2 * a, 1.0));
        }
    }
    if f.since_kick < 1.3 {
        let a = 1.0 - f.since_kick / 1.3;
        ring_h(vec3(0., 0.08, 0.), f.since_kick * 13.0, 28, 0., Color::new(1.0 * a, 0.45 * a, 0.1 * a, 1.0));
    }
    // Firelight pooling on the ground, flickering.
    let fl = 0.75 + 0.25 * (t * 9.0).sin() * (t * 5.3).cos();
    for i in 0..4 {
        ring_h(vec3(0., 0.06, 0.), 4.5 + i as f32 * 2.2 + f.bass * 0.4, if eco { 18 } else { 28 }, 0., Color::new(0.6 * fl * (1.0 - i as f32 * 0.2), 0.22 * fl * (1.0 - i as f32 * 0.2), 0.05, 1.0));
    }
    // Smoke wisps.
    for w in 0..cnt(eco, 3) {
        let mut last = vec3(0., 7.0, 0.);
        for j in 1..=8 {
            let y = 7.0 + j as f32 * 1.7;
            let p = vec3((y * 0.5 + t * 0.4 + w as f32 * 2.0).sin() * 0.9 + wind * (y - 7.0) * 0.25, y, (y * 0.4 + t * 0.3 + w as f32).cos() * 0.9);
            draw_line_3d(last, p, Color::new(0.30, 0.30, 0.33, 1.0) );
            last = p;
        }
    }
}

// ---------------------------------------------------------------- OCEAN (theme 8)
pub fn ocean(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    // Lighthouse with a sweeping beam.
    let base = vec3(-36.0, -1.0, -40.0);
    for k in 0..=4 {
        let y = k as f32 * 4.0;
        let r = 2.2 - 0.2 * k as f32;
        ring_h(base + vec3(0., y, 0.), r, 8, 0., if k % 2 == 0 { Color::new(0.9, 0.9, 0.9, 1.) } else { Color::new(0.8, 0.15, 0.12, 1.) });
    }
    for k in 0..8 {
        let a = TAU * k as f32 / 8.0;
        draw_line_3d(base + vec3(a.cos() * 2.2, 0., a.sin() * 2.2), base + vec3(a.cos() * 1.4, 16.0, a.sin() * 1.4), Color::new(0.7, 0.7, 0.72, 1.));
    }
    let lamp = base + vec3(0., 17.2, 0.);
    draw_cube_wires(lamp, vec3(2.0, 2.4, 2.0), Color::new(0.95, 0.85, 0.4, 1.));
    let ang = t * 0.6;
    for (dir, k) in [(0.0f32, 1.0f32), (std::f32::consts::PI, 0.45)] {
        for w in [-0.05f32, 0.0, 0.05] {
            let a = ang + dir + w;
            draw_line_3d(lamp, lamp + vec3(a.cos() * 70.0, -3.0, a.sin() * 70.0), Color::new(0.9 * k, 0.85 * k, 0.5 * k, 1.));
        }
    }
    spark(lamp, 1.0 + 0.3 * f.mid, Color::new(1.0, 0.95, 0.6, 1.));
    // Bobbing buoys with blinking lamps.
    for i in 0..cnt(eco, 6) {
        let u = i as u32 + 500;
        let (x, z) = ((h01(u, 1) - 0.5) * 70.0, (h01(u, 2) - 0.5) * 70.0);
        let y = crate::ocean_height(x, z, t, f.bass, f.mid);
        let p = vec3(x, y + 0.2, z);
        ring_h(p, 0.5, 8, 0., Color::new(0.9, 0.3, 0.1, 1.));
        ring_h(p + vec3(0., 0.5, 0.), 0.4, 8, 0., Color::new(0.9, 0.9, 0.9, 1.));
        draw_line_3d(p + vec3(0., 0.5, 0.), p + vec3(0., 1.9, 0.), Color::new(0.6, 0.6, 0.6, 1.));
        if ((t * 0.5 + h01(u, 3)) % 1.0) < 0.2 {
            spark(p + vec3(0., 2.0, 0.), 0.45, if i % 2 == 0 { Color::new(1., 0.2, 0.2, 1.) } else { Color::new(0.2, 1., 0.3, 1.) });
        }
    }
    // Foam on wave crests.
    let step = if eco { 14.0f32 } else { 9.0 };
    let n = (90.0 / step) as i32;
    for ix in 0..n {
        for iz in 0..n {
            let x = -45.0 + ix as f32 * step + (h01((ix * 31 + iz) as u32, 1) - 0.5) * 4.0;
            let z = -45.0 + iz as f32 * step + (h01((ix * 31 + iz) as u32, 2) - 0.5) * 4.0;
            let h = crate::ocean_height(x, z, t, f.bass, f.mid);
            if h > 1.1 {
                let p = vec3(x, h + 0.08, z);
                draw_line_3d(p - vec3(0.5, 0., 0.), p + vec3(0.5, 0., 0.2), Color::new(0.85, 0.95, 1.0, 1.));
                draw_line_3d(p - vec3(0.2, 0., 0.4), p + vec3(0.3, 0., 0.5), Color::new(0.7, 0.9, 1.0, 1.));
            }
        }
    }
}

// ---------------------------------------------------------------- FIREFLY FOREST (themes 9, 10)
pub fn firefly(f: &Frame, cyber: bool) {
    let t = f.t;
    let eco = f.eco;
    let (glow_a, glow_b) = if cyber { (Color::new(0.1, 1.0, 0.75, 1.), Color::new(0.9, 0.2, 1.0, 1.)) } else { (Color::new(0.35, 0.95, 0.75, 1.), Color::new(0.85, 1.0, 0.35, 1.)) };
    // Glowing mushroom clusters.
    for i in 0..cnt(eco, 8) {
        let u = i as u32 + 700;
        let z = (h01(u, 1) - 0.5) * 50.0;
        let mut x = (h01(u, 2) - 0.5) * 50.0;
        if (x - crate::forest_river_x(z)).abs() < 5.0 { x -= 8.0; }
        let g = vec3(x, crate::forest_ground(x, z), z);
        let pulse = 0.55 + 0.45 * (0.5 + 0.5 * (t * 1.2 + h01(u, 3) * TAU).sin()) + 0.25 * f.kick;
        for c in 0..3 {
            let off = vec3((h01(u, 4 + c) - 0.5) * 1.6, 0., (h01(u, 7 + c) - 0.5) * 1.6);
            let hgt = 0.6 + 0.5 * h01(u, 10 + c);
            let r = 0.35 + 0.25 * h01(u, 13 + c);
            let top = g + off + vec3(0., hgt, 0.);
            draw_line_3d(g + off, top, scale(glow_a, 0.35));
            let col = scale(mixc(glow_a, glow_b, h01(u, 16 + c)), pulse);
            ring_h(top, r, 8, 0., col);
            ring_h(top + vec3(0., 0.12, 0.), r * 0.55, 6, 0., col);
            for k in 0..6 {
                let a = TAU * k as f32 / 6.0;
                draw_line_3d(top + vec3(a.cos() * r, 0., a.sin() * r), top + vec3(0., 0.2, 0.), scale(col, 0.7));
            }
        }
    }
    // Moonbeams through the canopy.
    for i in 0..cnt(eco, 6) {
        let u = i as u32 + 900;
        let x = -24.0 + i as f32 * 9.0;
        let z = (h01(u, 1) - 0.5) * 40.0;
        for w in [-0.4f32, 0.0, 0.4] {
            draw_line_3d(vec3(x + 7.0 + w, 32.0, z - 4.0), vec3(x + w, crate::forest_ground(x, z), z), Color::new(0.05, 0.08, 0.14, 1.0));
        }
    }
    // Wandering fireflies: each follows its own lazy path and pulse; kicks send a wave of light outward.
    for i in 0..cnt(eco, 90) {
        let u = i as u32 + 1200;
        let hx = (h01(u, 1) - 0.5) * 70.0;
        let hz = (h01(u, 2) - 0.5) * 70.0;
        let sx = 0.31 * (0.7 + h01(u, 4));
        let sz = 0.27 * (0.7 + h01(u, 5));
        let p = vec3(
            hx + (t * sx + h01(u, 6) * TAU).sin() * 4.0 + (t * sx * 2.7 + h01(u, 7) * TAU).sin() * 1.2,
            crate::forest_ground(hx, hz) + 1.5 + h01(u, 3) * 8.0 + (t * 0.5 * (0.6 + h01(u, 8)) + h01(u, 9) * TAU).sin() * 1.5,
            hz + (t * sz + h01(u, 10) * TAU).cos() * 4.0 + (t * sz * 2.3 + h01(u, 11) * TAU).sin() * 1.2,
        );
        let pulse = (0.5 + 0.5 * (t * (1.0 + h01(u, 12) * 1.5) + h01(u, 13) * TAU).sin()).powi(3);
        let d = (p.x * p.x + p.z * p.z).sqrt();
        let wave = (1.0 - ((d - f.since_kick * 30.0) / 6.0).abs()).max(0.0) * (1.0 - f.since_kick / 1.5).max(0.0);
        let g = (0.3 + 0.7 * pulse + 0.4 * wave).min(1.2);
        let col = scale(mixc(glow_b, glow_a, h01(u, 14)), g);
        spark(p, 0.10 + 0.16 * g, col);
        if g > 0.85 { spark(p, 0.30, scale(col, 0.5)); }
    }
}
}
mod fx {
//! Shared frame context and small drawing helpers for the newer scene modules.
#![allow(dead_code)]

use crate::audio::BANDS;
use macroquad::prelude::*;
use std::f32::consts::TAU;

/// Everything a scene needs for one frame. `bass/mid/treble/spectrum` are the gentle,
/// smoothed values; `kick/snare/hat` are decaying hit pulses; `since_*` are seconds
/// since the last detected hit (used to run shockwaves without any per-hit state).
pub struct Frame {
    pub t: f32,
    pub dt: f32,
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub air: f32,
    pub kick: f32,
    pub snare: f32,
    pub hat: f32,
    pub energy: f32,
    pub beat_phase: f32,
    pub since_kick: f32,
    pub since_snare: f32,
    pub eco: bool,
    pub spectrum: [f32; BANDS],
}

/// Stable pseudo-random value in 0..1 from an index and a channel. No state, no allocation.
pub fn h01(i: u32, k: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B1) ^ k.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297A_2D39);
    x ^= x >> 15;
    (x >> 8) as f32 / 16_777_215.0
}

/// Item count: full in DETAIL, about a third in ECO (never below 1).
pub fn cnt(eco: bool, full: usize) -> usize {
    if eco { (full / 3).max(1) } else { full }
}

pub fn hsv(h: f32, s: f32, v: f32, a: f32) -> Color {
    let h = h.rem_euclid(1.0) * 6.0;
    let i = h.floor();
    let f = h - i;
    let p = v * (1. - s);
    let q = v * (1. - s * f);
    let t = v * (1. - s * (1. - f));
    let (r, g, b) = match i as i32 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color::new(r, g, b, a)
}
pub fn scale(c: Color, k: f32) -> Color {
    Color::new((c.r * k).min(1.), (c.g * k).min(1.), (c.b * k).min(1.), c.a)
}
pub fn alpha(c: Color, a: f32) -> Color { Color::new(c.r, c.g, c.b, a) }
pub fn mixc(a: Color, b: Color, t: f32) -> Color {
    let t = t.clamp(0., 1.);
    Color::new(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t)
}
/// Smooth 0..1 ease.
pub fn smooth(x: f32) -> f32 { let x = x.clamp(0., 1.); x * x * (3. - 2. * x) }

/// Horizontal circle of line segments.
pub fn ring_h(c: Vec3, r: f32, segs: usize, rot: f32, col: Color) {
    for i in 0..segs {
        let a = rot + TAU * i as f32 / segs as f32;
        let b = rot + TAU * (i + 1) as f32 / segs as f32;
        draw_line_3d(c + vec3(a.cos() * r, 0., a.sin() * r), c + vec3(b.cos() * r, 0., b.sin() * r), col);
    }
}
/// Horizontal arc from angle a0 to a1.
pub fn arc_h(c: Vec3, r: f32, a0: f32, a1: f32, segs: usize, col: Color) {
    for i in 0..segs {
        let a = a0 + (a1 - a0) * i as f32 / segs as f32;
        let b = a0 + (a1 - a0) * (i + 1) as f32 / segs as f32;
        draw_line_3d(c + vec3(a.cos() * r, 0., a.sin() * r), c + vec3(b.cos() * r, 0., b.sin() * r), col);
    }
}
/// Vertical circle in the XY plane (faces the camera when it looks along Z).
pub fn ring_xy(c: Vec3, r: f32, segs: usize, col: Color) {
    for i in 0..segs {
        let a = TAU * i as f32 / segs as f32;
        let b = TAU * (i + 1) as f32 / segs as f32;
        draw_line_3d(c + vec3(a.cos() * r, a.sin() * r, 0.), c + vec3(b.cos() * r, b.sin() * r, 0.), col);
    }
}
/// Tiny three-axis cross: the cheap stand-in for a glowing point.
pub fn spark(p: Vec3, s: f32, col: Color) {
    draw_line_3d(p - vec3(s, 0., 0.), p + vec3(s, 0., 0.), col);
    draw_line_3d(p - vec3(0., s, 0.), p + vec3(0., s, 0.), col);
    draw_line_3d(p - vec3(0., 0., s), p + vec3(0., 0., s), col);
}
pub fn poly(pts: &[Vec3], col: Color) {
    for w in pts.windows(2) { draw_line_3d(w[0], w[1], col); }
}
/// Deterministic jagged bolt between two points; `seed` changes the shape.
pub fn bolt(a: Vec3, b: Vec3, seed: f32, jag: f32, col: Color) {
    let mut last = a;
    for k in 1..=8 {
        let f = k as f32 / 8.;
        let mut p = a.lerp(b, f);
        if k < 8 {
            p += vec3((seed * 12.9 + k as f32 * 7.1).sin(), (seed * 4.1 + k as f32 * 3.3).sin() * 0.6, (seed * 8.7 + k as f32 * 5.9).sin()) * jag;
        }
        draw_line_3d(last, p, col);
        last = p;
    }
}
/// Wireframe octahedron rotated about Y.
pub fn octa(c: Vec3, s: f32, rot: f32, col: Color) {
    let (sn, cs) = rot.sin_cos();
    let v = |x: f32, y: f32, z: f32| c + vec3(x * cs - z * sn, y, x * sn + z * cs) * s;
    let (px, nx, py, ny, pz, nz) = (v(1., 0., 0.), v(-1., 0., 0.), v(0., 1.4, 0.), v(0., -1.4, 0.), v(0., 0., 1.), v(0., 0., -1.));
    for &(a, b) in &[(px, pz), (pz, nx), (nx, nz), (nz, px)] {
        draw_line_3d(a, b, col);
        draw_line_3d(py, a, col);
        draw_line_3d(ny, a, col);
    }
}
}
mod synth {
//! Rebuilt scenes: SYNTHWAVE (theme 1) and BEAT SYNC (theme 2).
//!
//! Both are procedural line/cube scenes with no per-frame allocation, with fixed
//! geometry budgets (listed beside each loop) and ECO versions at roughly a quarter
//! of the DETAIL cost.
#![allow(dead_code)]

use crate::audio::BANDS;
use crate::fx::*;
use macroquad::prelude::*;
use std::f32::consts::TAU;

fn bg() -> Color { Color::new(0.02, 0.0, 0.06, 1.0) }

// ============================================================================
// SYNTHWAVE
// ============================================================================

pub struct Synth {
    pub scroll: f32,
}

impl Synth {
    pub fn new() -> Self { Self { scroll: 0. } }

    /// Drive speed follows musical energy; bass adds a little surge.
    pub fn update(&mut self, f: &Frame) {
        self.scroll += f.dt * (7.0 + 13.0 * f.energy + 5.0 * f.bass);
    }

    fn terrain(x: f32, zw: f32, t: f32, bass: f32) -> f32 {
        let side = smooth((x.abs() - 12.0) / 28.0);
        let n = 0.5 + 0.25 * (zw * 0.05 + x * 0.21).sin() + 0.2 * (zw * 0.11 - x * 0.13 + 1.7).sin() + 0.15 * (zw * 0.23 + x * 0.31).sin();
        side * (2.5 + 9.0 * n.abs()) + bass * 0.7 * (x * 0.2 + zw * 0.15 + t * 1.5).sin() * (0.35 + side)
    }

    pub fn draw(&self, f: &Frame, grid: bool) {
        let t = f.t;
        let eco = f.eco;
        let m = 0.5 - 0.5 * (t * 0.035).cos(); // slow vaporwave drift: 0 = sunset, 1 = midnight
        let grid_c = hsv(0.88 - 0.36 * m, 0.85, 1.0, 1.0);
        let road_c = hsv(0.52 + 0.30 * m, 0.8, 1.0, 1.0);
        let sun_top = mixc(Color::new(1.0, 0.85, 0.2, 1.), Color::new(0.2, 0.95, 1.0, 1.), m);
        let sun_bot = mixc(Color::new(1.0, 0.15, 0.55, 1.), Color::new(0.65, 0.2, 1.0, 1.), m);
        let sc = self.scroll;

        // ---- ground grid: DETAIL ~1450 lines, ECO ~370 -------------------------------------------
        let (sp, step_x, nr, ncol) = if eco { (10.0f32, 12.0f32, 19usize, 10usize) } else { (5.0, 6.0, 38, 20) };
        let frac = sc % sp;
        let mut prev = [0f32; 24];
        for i in 0..=nr {
            let zs = 38.0 - i as f32 * sp + frac;
            let zw = zs - sc;
            let fog = ((38.0 - zs) / 230.0).clamp(0., 1.).powf(1.3);
            let mut pw = 0.0;
            if f.since_kick < 2.0 {
                let d = (zs - (38.0 - f.since_kick * 75.0)) / 7.0;
                pw = (-d * d).exp() * (1.0 - f.since_kick * 0.5) * 0.55; // kick wave rolls toward the horizon
            }
            let mut cur = [0f32; 24];
            for c in 0..=ncol {
                let x = -60.0 + c as f32 * step_x;
                cur[c] = Self::terrain(x, zw, t, f.bass) + pw * 0.9;
            }
            if grid {
                let rc = mixc(scale(grid_c, 0.9 + pw), bg(), fog);
                for c in 0..ncol {
                    let x = -60.0 + c as f32 * step_x;
                    draw_line_3d(vec3(x, cur[c], zs), vec3(x + step_x, cur[c + 1], zs), rc);
                }
                if i > 0 {
                    for c in 0..=ncol {
                        let x = -60.0 + c as f32 * step_x;
                        draw_line_3d(vec3(x, prev[c], zs + sp), vec3(x, cur[c], zs), rc);
                    }
                }
            }
            if i > 0 {
                // road edges and centre dashes
                let ec = mixc(scale(road_c, 0.9 + pw), bg(), fog);
                for s in [-12.0f32, 12.0] {
                    draw_line_3d(vec3(s, 0.05, zs + sp), vec3(s, 0.05, zs), ec);
                }
                draw_line_3d(vec3(0., 0.05, zs + sp * 0.8), vec3(0., 0.05, zs + sp * 0.3), mixc(scale(road_c, 0.7), bg(), fog));
            }
            prev = cur;
        }

        // ---- distant mountains (solid silhouette so the sun sits behind them) -------------------
        let mstep = if eco { 12.0f32 } else { 6.0 };
        let mn = (300.0 / mstep) as usize;
        for layer in 0..2 {
            let (z, hs, col) = if layer == 0 { (-158.0f32, 1.0f32, Color::new(0.11, 0.02, 0.22, 1.)) } else { (-142.0, 0.55, Color::new(0.05, 0.0, 0.12, 1.)) };
            for i in 0..mn {
                let x = -150.0 + i as f32 * mstep;
                let xl = x + layer as f32 * 19.0;
                let h = (10.0 + 16.0 * (xl * 0.045 + 1.3).sin().abs() * (0.6 + 0.4 * (xl * 0.13).sin()) + 4.0 * (xl * 0.31).sin().abs()) * hs;
                draw_cube(vec3(x, h * 0.5, z), vec3(mstep, h, 3.0), None, col);
                if layer == 0 {
                    let h2 = (10.0 + 16.0 * ((xl + mstep) * 0.045 + 1.3).sin().abs() * (0.6 + 0.4 * ((xl + mstep) * 0.13).sin()) + 4.0 * ((xl + mstep) * 0.31).sin().abs()) * hs;
                    draw_line_3d(vec3(x, h, z + 1.6), vec3(x + mstep, h2, z + 1.6), scale(road_c, 0.75));
                }
            }
        }

        // ---- the sun: 22 striped slices with gaps that breathe with the bass --------------------------
        let sun_c = vec3(0., 36.0, -175.0);
        let sr = 60.0 * (1.0 + 0.03 * f.bass);
        let slices = 22usize;
        for i in 0..slices {
            let y0 = -sr + 2.0 * sr * i as f32 / slices as f32;
            let hh = 2.0 * sr / slices as f32;
            let gap = ((10.0 - i as f32) / 10.0).clamp(0., 1.) * (0.65 - 0.2 * f.bass);
            let h = hh * (1.0 - gap);
            let yc = y0 + hh * (1.0 - gap) * 0.5;
            let dy = (yc).abs().min(sr * 0.999);
            let w = 2.0 * (sr * sr - dy * dy).sqrt();
            let col = mixc(sun_bot, sun_top, (yc + sr) / (2.0 * sr));
            draw_cube(sun_c + vec3(0., yc, 0.), vec3(w, h, 2.0), None, col);
        }
        let halo = if eco { 1 } else { 3 };
        for k in 0..halo {
            let r = sr * (1.1 + 0.16 * k as f32) + f.bass * 1.2;
            ring_xy(sun_c + vec3(0., 0., 1.5), r, if eco { 32 } else { 56 }, alpha(mixc(sun_bot, sun_top, 0.5), 0.35 - 0.1 * k as f32));
        }

        // ---- horizon glow bands ---------------------------------------------------------------------
        for k in 0..6 {
            let y = 0.4 + k as f32 * 0.9;
            let c = alpha(mixc(sun_bot, sun_top, k as f32 / 6.0), 0.5 - k as f32 * 0.07);
            draw_line_3d(vec3(-150., y, -150.0), vec3(150., y, -150.0), c);
        }

        // ---- stars: 70 DETAIL / 23 ECO; twinkle with treble -------------------------------------------
        for i in 0..cnt(eco, 70) {
            let az = (h01(i as u32, 1) - 0.5) * 2.4;
            let el = 0.10 + h01(i as u32, 2) * 0.65;
            let p = vec3(az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()) * 200.0;
            let tw = 0.4 + 0.6 * (0.5 + 0.5 * (t * (0.7 + h01(i as u32, 3) * 2.0) + h01(i as u32, 4) * TAU).sin());
            let c = hsv(0.75 + 0.2 * h01(i as u32, 5), 0.3, 0.45 + 0.55 * tw * (0.6 + f.treble), 1.0);
            spark(p, 0.35 + 0.35 * tw, c);
        }

        // ---- neon pylons on a 20-unit lattice: 20 lines ----------------------------------------------
        let k = (sc / 20.0).floor() as i32;
        let fr = sc % 20.0;
        for j in 0..10 {
            let zs = 38.0 - j as f32 * 20.0 + fr;
            let fog = ((38.0 - zs) / 230.0).clamp(0., 1.);
            let c = mixc(scale(if (j + k).rem_euclid(2) == 0 { grid_c } else { road_c }, 0.8 + 0.25 * f.kick), bg(), fog);
            for side in [-1.0f32, 1.0] {
                let x = side * 17.0;
                let base = vec3(x, Self::terrain(x, zs - sc, t, f.bass), zs);
                let top = base + vec3(0., 7.5, 0.);
                draw_line_3d(base, top, c);
                draw_line_3d(top, top + vec3(-side * 2.2, 0., 0.), c);
                spark(top, 0.4 + 0.2 * f.kick, c);
            }
        }

        // ---- floating wireframe pyramids on the hills (4 DETAIL / 1 ECO), recycled every 160 units -----
        for j in 0..cnt(eco, 4) {
            let zs = -150.0 + (j as f32 * 40.0 + sc) % 160.0;
            let cycle = ((j as f32 * 40.0 + sc) / 160.0).floor() as u32;
            for side in [-1.0f32, 1.0] {
                let x = side * (30.0 + 8.0 * h01(cycle * 4 + j as u32, 9));
                let y = Self::terrain(x, zs - sc, t, f.bass) + 6.0 + (t * 0.8 + j as f32).sin();
                octa(vec3(x, y, zs), 2.4 + f.mid, t * 0.5 + j as f32, mixc(road_c, bg(), ((38.0 - zs) / 230.0).clamp(0., 1.)));
            }
        }

        // ---- light trails: 8 cars x 5 segments (ECO 2 cars) ---------------------------------------------
        for i in 0..cnt(eco, 8) {
            let lane = [-9.0f32, -4.5, 4.5, 9.0][i % 4];
            let spd = 24.0 + h01(i as u32, 1) * 22.0;
            let ph = h01(i as u32, 2) * 200.0;
            let (head, dir, col) = if lane < 0.0 {
                (-160.0 + (t * spd + ph) % 200.0, 1.0f32, Color::new(0.75, 0.95, 1.0, 1.))
            } else {
                (40.0 - (t * spd + ph) % 200.0, -1.0, Color::new(1.0, 0.15, 0.3, 1.))
            };
            let mut last = vec3(lane, 0.25, head);
            for s in 1..=5 {
                let p = vec3(lane, 0.25, head - dir * 3.2 * s as f32);
                let fog = ((38.0 - p.z) / 230.0).clamp(0., 1.);
                draw_line_3d(last, p, mixc(scale(col, 1.0 - s as f32 * 0.17), bg(), fog));
                last = p;
            }
        }
    }
}

// ============================================================================
// BEAT SYNC: a spectral tunnel
// ============================================================================

pub const SEC: usize = 24; // vertices per ring (12 frequency sectors, mirrored for kaleidoscope symmetry)
pub const RINGS: usize = 36;
const INTERVAL: f32 = 0.09;

pub struct Tunnel {
    hist: [[f32; SEC]; RINGS],
    timer: f32,
    spin: f32,
    shifts: i64,
}

impl Tunnel {
    pub fn new() -> Self { Self { hist: [[0.; SEC]; RINGS], timer: 0., spin: 0., shifts: 0 } }

    /// Centre line of the tunnel: a lazy snake the camera follows.
    pub fn path(z: f32, t: f32) -> (f32, f32) { ((z * 0.02 + t * 0.3).sin() * 4.0, (z * 0.025 + t * 0.23).cos() * 2.5) }

    /// Push the current spectrum in as the newest ring; older rings recede into the distance.
    pub fn update(&mut self, f: &Frame) {
        self.spin += f.dt * (0.10 + 0.22 * f.energy);
        self.timer += f.dt;
        let mut n = 0;
        while self.timer >= INTERVAL && n < 3 {
            self.timer -= INTERVAL;
            n += 1;
            self.shifts += 1;
            self.hist.rotate_right(1);
            let mut row = [0f32; SEC];
            for j in 0..12 {
                let a = j * 5;
                let b = (a + 6).min(BANDS);
                let mut s = 0.;
                for k in a..b { s += f.spectrum[k]; }
                let v = s / (b - a) as f32;
                row[j] = v;
                row[SEC - 1 - j] = v;
            }
            self.hist[0] = row;
        }
        if self.timer > INTERVAL { self.timer = INTERVAL * 0.99; }
    }

    /// DETAIL: 36 rings x 24 sectors (~1700 lines). ECO: every other ring, every other sector (~300).
    pub fn draw(&self, f: &Frame) {
        let eco = f.eco;
        let t = f.t;
        let sp = 4.0f32;
        let adv = (self.timer / INTERVAL).min(1.0);
        let hue0 = (t * 0.02).fract();
        let accent = hue0 + 0.5; // complementary accent colour
        let z0 = 4.0f32;
        let r_base = 7.0 + f.bass * 1.3;
        let sec_step = if eco { 2 } else { 1 };
        let mut prev = [Vec3::ZERO; SEC];
        let mut have_prev = false;
        for i in 0..RINGS {
            if eco && (i as i64 + self.shifts).rem_euclid(2) != 0 { continue; }
            let z = z0 - (i as f32 + adv) * sp;
            let (cx, cy) = Self::path(z, t);
            let id = (self.shifts - i as i64) as f32;
            let ang = self.spin + id * 0.06;
            let depth = i as f32 / RINGS as f32;
            let fade = (1.0 - depth).powf(0.8).max(0.06);
            let mut cur = [Vec3::ZERO; SEC];
            for k in 0..SEC {
                let h = self.hist[i][k];
                let r = r_base + 4.2 * h * (1.0 - depth * 0.3);
                let th = TAU * k as f32 / SEC as f32 + ang;
                cur[k] = vec3(cx + r * th.cos(), cy + r * th.sin(), z);
            }
            let mut k = 0;
            while k < SEC {
                let k2 = (k + sec_step) % SEC;
                let h = self.hist[i][k];
                let c = hsv(hue0 + id * 0.012 + k as f32 / SEC as f32 * 0.12, 0.9, (0.45 + 0.7 * h).min(1.0) * fade, 1.0);
                draw_line_3d(cur[k], cur[k2], c);
                if have_prev {
                    let ribbon = k % 6 == 3;
                    let lc = if ribbon { hsv(accent + id * 0.006, 0.7, (0.55 + 0.45 * h) * fade, 1.0) } else { scale(c, 0.7) };
                    draw_line_3d(prev[k], cur[k], lc);
                    if ribbon {
                        draw_line_3d(prev[k] + vec3(0.12, 0.12, 0.), cur[k] + vec3(0.12, 0.12, 0.), lc);
                    }
                }
                k += sec_step;
            }
            prev = cur;
            have_prev = true;
        }

        // ---- beat shockwaves: a bright ring for each kick, a thin one for each snare ---------------------
        let zp = |z: f32| Self::path(z, t);
        if f.since_kick < 1.6 {
            let z = z0 - f.since_kick * 45.0;
            let (cx, cy) = zp(z);
            let a = 1.0 - f.since_kick / 1.6;
            for dz in [0.0f32, 0.35] {
                let c = Color::new(0.8 * a, 0.95 * a, 1.0 * a, 1.0);
                for k in 0..SEC {
                    let th0 = TAU * k as f32 / SEC as f32;
                    let th1 = TAU * (k + 1) as f32 / SEC as f32;
                    let r = r_base + 1.2;
                    draw_line_3d(vec3(cx + r * th0.cos(), cy + r * th0.sin(), z - dz), vec3(cx + r * th1.cos(), cy + r * th1.sin(), z - dz), c);
                }
            }
        }
        if f.since_snare < 1.2 {
            let z = z0 - f.since_snare * 60.0;
            let (cx, cy) = zp(z);
            let a = 1.0 - f.since_snare / 1.2;
            let c = hsv(accent, 0.8, a, 1.0);
            for k in 0..12 {
                let th0 = TAU * k as f32 / 12.0 + self.spin;
                let th1 = TAU * (k + 1) as f32 / 12.0 + self.spin;
                let r = r_base * 0.8;
                draw_line_3d(vec3(cx + r * th0.cos(), cy + r * th0.sin(), z), vec3(cx + r * th1.cos(), cy + r * th1.sin(), z), c);
            }
        }

        // ---- bass prisms orbiting the axis -----------------------------------------------------------------
        for j in 0..cnt(eco, 6) {
            let z = -22.0 - j as f32 * 9.0;
            let (cx, cy) = zp(z);
            let a = self.spin * 1.5 + j as f32 * 1.05;
            let r = r_base * 0.45;
            let s = 1.0 + f.bass * 1.6;
            draw_cube_wires(vec3(cx + r * a.cos(), cy + r * a.sin(), z), vec3(s, s, s), hsv(accent + j as f32 * 0.04, 0.7, 0.9, 1.0));
        }

        // ---- particle field streaming down the tunnel -----------------------------------------------------
        for i in 0..cnt(eco, 90) {
            let u = i as u32;
            let z = z0 - 2.0 - (t * (14.0 + h01(u, 1) * 20.0) + h01(u, 2) * 300.0) % 140.0;
            let (cx, cy) = zp(z);
            let a = h01(u, 3) * TAU + t * 0.2 * (h01(u, 4) - 0.5);
            let r = (0.2 + 0.7 * h01(u, 5)) * r_base;
            let fade = (1.0 - (z0 - z) / 140.0).clamp(0., 1.);
            spark(vec3(cx + r * a.cos(), cy + r * a.sin(), z), 0.08 + f.treble * 0.14, hsv(hue0 + 0.15 * h01(u, 6), 0.3, 0.3 + 0.7 * fade, 1.0));
        }
    }
}
}
mod toxic {
//! TOXIC (theme 0), rebuilt: an abandoned, still-living nuclear facility.
//!
//! Layout (Y is up, ground at 0, everything within about +-36 units of the reactor):
//!   reactor with pulsing core, gantry and control rods at the origin; a glowing coolant moat
//!   around it; four pipelines (one overhead) feeding three storage tanks; a barrel yard with
//!   leaking barrels; contaminated ground with glowing cracks and toxic pools; four pylons with
//!   sagging cables carrying current pulses; a sweeping searchlight; hazard signs; vapour,
//!   smoke and sparks; and a skyline of cooling towers in the distance.
//!
//! Musical mapping (all through the gentle Frame values, so nothing strobes):
//!   bass/kick -> reactor core pulse, ground shockwave, tank liquid;  mid -> overall lighting
//!   and pipe flow;  treble -> sparks, bubbles, radiation flecks;  snare -> electric arcs and
//!   beacon brightness;  energy -> liquid flow speed.
//!
//! Geometry budget: roughly 1,900 line segments in DETAIL and 500 in ECO (counted from the loops
//! by hand, not measured on a real GPU).
#![allow(dead_code)]

use crate::fx::*;
use macroquad::prelude::*;
use std::f32::consts::{PI, TAU};

fn acid() -> Color { Color::new(0.35, 1.0, 0.10, 1.0) }
fn sick() -> Color { Color::new(0.95, 0.90, 0.12, 1.0) }
fn concrete() -> Color { Color::new(0.34, 0.42, 0.22, 1.0) }
fn steel() -> Color { Color::new(0.55, 0.68, 0.58, 1.0) }
fn rust() -> Color { Color::new(0.42, 0.34, 0.12, 1.0) }

/// Distance fade so the far parts of the facility sink into the dark.
fn fog(c: Color, p: Vec3) -> Color {
    let d = (p.x * p.x + p.z * p.z).sqrt();
    scale(c, (1.0 - d / 110.0).clamp(0.3, 1.0))
}

/// Circle of line segments around an arbitrary axis.
fn loop_axis(c: Vec3, axis: Vec3, r: f32, segs: usize, col: Color) {
    let a = axis.normalize_or_zero();
    let helper = if a.y.abs() > 0.9 { vec3(1., 0., 0.) } else { vec3(0., 1., 0.) };
    let u = a.cross(helper).normalize_or_zero();
    let v = a.cross(u);
    for i in 0..segs {
        let t0 = TAU * i as f32 / segs as f32;
        let t1 = TAU * (i + 1) as f32 / segs as f32;
        draw_line_3d(c + (u * t0.cos() + v * t0.sin()) * r, c + (u * t1.cos() + v * t1.sin()) * r, col);
    }
}

/// A drum: three hoops and four struts around `axis`, centred on `c`.
fn barrel(c: Vec3, axis: Vec3, col: Color, segs: usize) {
    let a = axis.normalize_or_zero();
    let helper = if a.y.abs() > 0.9 { vec3(1., 0., 0.) } else { vec3(0., 1., 0.) };
    let u = a.cross(helper).normalize_or_zero() * 0.62;
    let (lo, hi) = (c - a * 0.75, c + a * 0.75);
    for p in [lo, c, hi] { loop_axis(p, a, 0.62, segs, col); }
    for k in 0..4 {
        let ang = k as f32 * TAU / 4.0;
        let off = u * ang.cos() + a.cross(u) * ang.sin();
        draw_line_3d(lo + off, hi + off, col);
    }
}

/// Square-section pipe with flanges.
fn duct(a: Vec3, b: Vec3, r: f32, col: Color, flange_step: f32) {
    let d = b - a;
    let len = d.length();
    if len < 0.01 { return; }
    let dir = d / len;
    let helper = if dir.y.abs() > 0.9 { vec3(1., 0., 0.) } else { vec3(0., 1., 0.) };
    let u = dir.cross(helper).normalize_or_zero() * r;
    let v = dir.cross(u).normalize_or_zero() * r;
    let corners = [u + v, u - v, -u - v, -u + v];
    for o in corners { draw_line_3d(a + o, b + o, col); }
    let n = (len / flange_step).floor() as i32;
    for i in 0..=n {
        let p = a + dir * (flange_step * i as f32);
        for k in 0..4 { draw_line_3d(p + corners[k] * 1.3, p + corners[(k + 1) % 4] * 1.3, scale(col, 1.15)); }
    }
}

/// Bright dashes of liquid travelling along a pipe.
fn flow(a: Vec3, b: Vec3, t: f32, speed: f32, phase: f32, col: Color) {
    for k in 0..2 {
        let h = (t * speed + phase + k as f32 * 0.5).fract();
        draw_line_3d(a.lerp(b, h), a.lerp(b, (h + 0.1).min(1.0)), col);
    }
}

/// Point on a drooping, gently swaying cable.
fn cable_pt(a: Vec3, b: Vec3, u: f32, t: f32, sag: f32) -> Vec3 {
    let mut p = a.lerp(b, u);
    p.y -= sag * 4.0 * u * (1.0 - u);
    p.x += (t * 0.9 + u * 5.0).sin() * 0.12 * 4.0 * u * (1.0 - u);
    p
}

/// Irregular pool outline with two glowing contours.
fn pool(c: Vec3, r: f32, t: f32, seed: f32, glow: f32, eco: bool) {
    let n = if eco { 10 } else { 14 };
    for (k, sc) in [1.0f32, 0.62, 0.3].iter().enumerate() {
        if eco && k == 2 { break; }
        let mut last = Vec3::ZERO;
        for i in 0..=n {
            let a = TAU * i as f32 / n as f32 + k as f32 * 0.4 + t * 0.05;
            let w = 1.0 + 0.18 * (a * 3.0 + seed + t * 0.8).sin() + 0.08 * (a * 5.0 - seed).sin();
            let p = c + vec3(a.cos(), 0.0, a.sin()) * (r * sc * w);
            if i > 0 { draw_line_3d(last, p, scale(acid(), (0.25 + 0.6 * glow) * (1.0 - 0.25 * k as f32))); }
            last = p;
        }
    }
    // ripple from a drip point
    let age = (t * 0.55 + seed).fract();
    ring_h(c + vec3(r * 0.3, 0.02, 0.0), age * r * 0.7, 10, 0., scale(acid(), (1.0 - age) * (0.3 + 0.5 * glow)));
}

/// Radiation trefoil on a post, drawn in the XY plane.
fn trefoil(p: Vec3, r: f32, col: Color) {
    ring_xy(p, r, 20, col);
    ring_xy(p, r * 0.18, 8, col);
    for k in 0..3 {
        let a0 = TAU * k as f32 / 3.0 + 0.3;
        let mut last = Vec3::ZERO;
        for j in 0..=5 {
            let a = a0 + (PI / 3.0) * j as f32 / 5.0;
            let q = p + vec3(a.cos(), a.sin(), 0.) * (r * 0.8);
            if j > 0 { draw_line_3d(last, q, col); }
            last = q;
        }
        for rr in [0.3f32, 0.8] { draw_line_3d(p + vec3(a0.cos(), a0.sin(), 0.) * (r * rr), p + vec3((a0 + PI / 3.0).cos(), (a0 + PI / 3.0).sin(), 0.) * (r * rr), col); }
    }
}

pub fn draw(f: &Frame) {
    let t = f.t;
    let eco = f.eco;
    let core = (f.bass * 0.7 + f.kick * 0.3).clamp(0., 1.);
    let idle = 0.5 + 0.5 * (t * 1.1).sin();
    let glow = (0.45 + 0.40 * core + 0.15 * idle).min(1.0); // overall radioactive glow
    let lum = 0.78 + 0.22 * f.mid; // lighting follows the mids
    let con = scale(concrete(), lum);
    let st = scale(steel(), lum);
    let seg = if eco { 12 } else { 16 };
    let flow_speed = 0.18 + 0.30 * f.energy;

    // ---------------------------------------------------------------- ground (contaminated slabs)
    let ext = 36.0f32;
    let gs = if eco { 12.0f32 } else { 6.0 };
    let n = (ext * 2.0 / gs) as i32;
    let wave_r = f.since_kick * 24.0;
    let wave_a = (1.0 - f.since_kick / 1.4).max(0.0);
    for line in 0..=n {
        let c = -ext + line as f32 * gs;
        for sgm in 0..n {
            let a = -ext + sgm as f32 * gs;
            let m = a + gs * 0.5;
            for (p0, p1, mid) in [
                (vec3(a, 0., c), vec3(a + gs, 0., c), vec3(m, 0., c)),
                (vec3(c, 0., a), vec3(c, 0., a + gs), vec3(c, 0., m)),
            ] {
                let d = (mid.x * mid.x + mid.z * mid.z).sqrt();
                let near = (1.0 - d / 24.0).clamp(0., 1.).powi(2);
                let wave = if f.since_kick < 1.4 { (-((d - wave_r) / 2.5).powi(2)).exp() * wave_a } else { 0.0 };
                let k = (near * glow * 0.75 + wave * 0.9).min(1.0);
                draw_line_3d(p0, p1, mixc(fog(con, mid), acid(), k));
            }
        }
    }
    // glowing cracks radiating from the reactor
    for i in 0..cnt(eco, 7) {
        let a0 = i as f32 * TAU / 7.0 + 0.3;
        let mut last = vec3(a0.cos() * 5.0, 0.03, a0.sin() * 5.0);
        for k in 1..=6 {
            let r = 5.0 + k as f32 * 3.2;
            let j = (k as f32 * 2.7 + i as f32).sin() * 0.9;
            let p = vec3(a0.cos() * r - a0.sin() * j, 0.03, a0.sin() * r + a0.cos() * j);
            let pulse = 0.55 + 0.45 * (t * 2.0 + k as f32 + i as f32).sin();
            draw_line_3d(last, p, scale(acid(), (0.35 + 0.55 * glow * pulse) * (1.0 - k as f32 * 0.1)));
            last = p;
        }
    }
    // coolant moat with liquid flowing around the reactor
    ring_h(vec3(0., 0.03, 0.), 8.2, if eco { 20 } else { 32 }, 0., scale(acid(), 0.35 + 0.4 * glow));
    ring_h(vec3(0., 0.03, 0.), 9.2, if eco { 20 } else { 32 }, 0., scale(acid(), 0.35 + 0.4 * glow));
    for i in 0..cnt(eco, 8) {
        let a = t * (0.35 + f.energy * 0.6) + i as f32 * TAU / 8.0;
        arc_h(vec3(0., 0.05, 0.), 8.7, a, a + 0.35, 4, mixc(acid(), sick(), 0.5 + 0.5 * core));
    }
    // toxic pools
    for (i, &(x, z, r)) in [(-9.0f32, -17.0f32, 3.0f32), (5.0, -18.0, 2.4), (22.0, 3.0, 2.6), (-8.0, 8.0, 2.0)].iter().enumerate() {
        if eco && i >= 2 { break; }
        pool(vec3(x, 0.03, z), r, t, i as f32 * 1.7, glow, eco);
        for b in 0..(if eco { 1 } else { 3 }) {
            let age = (t * 0.45 + h01(i as u32 * 7 + b, 3)).fract();
            let p = vec3(x + (h01(i as u32 * 7 + b, 4) - 0.5) * r, 0.1 + age * 1.4, z + (h01(i as u32 * 7 + b, 5) - 0.5) * r);
            spark(p, (0.06 + f.treble * 0.08) * (1.0 - age * 0.5), scale(acid(), 0.9));
        }
    }

    // ---------------------------------------------------------------- reactor
    for k in 0..=8 {
        let y = k as f32 * 1.1;
        let c = if k == 0 || k == 8 { scale(st, 1.2) } else { st };
        ring_h(vec3(0., y, 0.), 4.5, seg, 0., c);
    }
    for k in 0..cnt(eco, 12) {
        let a = TAU * k as f32 / cnt(eco, 12) as f32;
        draw_line_3d(vec3(a.cos() * 4.5, 0., a.sin() * 4.5), vec3(a.cos() * 4.5, 8.8, a.sin() * 4.5), scale(st, 0.8));
    }
    for m in 0..cnt(eco, 8) {
        let a = TAU * m as f32 / cnt(eco, 8) as f32;
        let mut last = vec3(a.cos() * 4.5, 8.8, a.sin() * 4.5);
        for j in 1..=4 {
            let phi = j as f32 * (PI / 2.0) / 4.0;
            let p = vec3(a.cos() * 4.5 * phi.cos(), 8.8 + 2.5 * phi.sin(), a.sin() * 4.5 * phi.cos());
            draw_line_3d(last, p, scale(st, 0.9));
            last = p;
        }
    }
    for j in 1..=3 {
        let phi = j as f32 * (PI / 2.0) / 4.0;
        ring_h(vec3(0., 8.8 + 2.5 * phi.sin(), 0.), 4.5 * phi.cos(), seg, 0., st);
    }
    // glowing inspection slits
    for k in 0..cnt(eco, 6) {
        let a = TAU * k as f32 / cnt(eco, 6) as f32 + 0.4;
        let (p0, p1) = (vec3((a - 0.18).cos() * 4.55, 2.5, (a - 0.18).sin() * 4.55), vec3((a + 0.18).cos() * 4.55, 2.5, (a + 0.18).sin() * 4.55));
        draw_line_3d(p0, p1, scale(acid(), 0.5 + 0.5 * glow));
        draw_line_3d(p0 + vec3(0., 0.25, 0.), p1 + vec3(0., 0.25, 0.), scale(sick(), 0.4 + 0.4 * glow));
    }
    // the core: three orthogonal rings, a crystal, expanding halos and moving control rods
    let cc = vec3(0., 4.4, 0.);
    let cr = 1.1 + 0.7 * core;
    let cc_col = scale(mixc(acid(), sick(), core), 0.55 + 0.45 * glow);
    ring_h(cc, cr, 14, t * 0.6, cc_col);
    ring_xy(cc, cr, 14, cc_col);
    loop_axis(cc, vec3(1., 0., 0.), cr, 14, cc_col);
    octa(cc, 0.8 + 0.5 * core, t * 0.7, scale(sick(), 0.5 + 0.5 * glow));
    for h in 0..cnt(eco, 3) {
        let age = (t * 0.5 + h as f32 / 3.0).fract();
        ring_h(cc, 1.5 + age * 3.2, 16, 0., scale(acid(), (1.0 - age) * 0.6 * glow));
    }
    for r in 0..cnt(eco, 8) {
        let a = TAU * r as f32 / cnt(eco, 8) as f32;
        let top = 8.0 - 2.5 * (0.5 + 0.5 * (t * 0.6 + r as f32).sin()) + core * 0.6;
        draw_line_3d(vec3(a.cos() * 2.2, 1.0, a.sin() * 2.2), vec3(a.cos() * 2.2, top, a.sin() * 2.2), scale(st, 1.1));
        spark(vec3(a.cos() * 2.2, top, a.sin() * 2.2), 0.12, scale(acid(), 0.7));
    }
    // gantry with hazard stripes, rails and rotating beacons
    for i in 0..16 {
        let (a0, a1) = (TAU * i as f32 / 16.0, TAU * (i + 1) as f32 / 16.0);
        let c = if i % 2 == 0 { sick() } else { Color::new(0.12, 0.12, 0.08, 1.0) };
        draw_line_3d(vec3(a0.cos() * 6.0, 5.4, a0.sin() * 6.0), vec3(a1.cos() * 6.0, 5.4, a1.sin() * 6.0), scale(c, lum));
    }
    ring_h(vec3(0., 6.4, 0.), 6.0, 16, 0., st);
    for i in 0..cnt(eco, 8) {
        let a = TAU * i as f32 / cnt(eco, 8) as f32;
        draw_line_3d(vec3(a.cos() * 6.0, 5.4, a.sin() * 6.0), vec3(a.cos() * 6.0, 6.4, a.sin() * 6.0), st);
    }
    for i in 0..4 {
        let a = TAU * i as f32 / 4.0 + PI / 4.0;
        let p = vec3(a.cos() * 6.0, 7.0, a.sin() * 6.0);
        let k = 0.35 + 0.45 * f.snare; // steady glow, brighter on snare, never a strobe
        let b = t * 2.5 + i as f32;
        spark(p, 0.35, Color::new(1.0 * k, 0.55 * k, 0.05 * k, 1.0));
        draw_line_3d(p, p + vec3(b.cos() * 2.2, -0.3, b.sin() * 2.2), Color::new(0.9 * k, 0.45 * k, 0.04 * k, 1.0));
    }

    // ---------------------------------------------------------------- storage tanks
    let tanks = [vec3(15., 0., -9.), vec3(-17., 0., -6.), vec3(13., 0., 14.)];
    let ntank = if eco { 2 } else { 3 };
    for (i, &tp) in tanks.iter().take(ntank).enumerate() {
        let ts = if eco { 8 } else { 12 };
        for k in 0..=6 {
            let y = k as f32 * 1.25;
            ring_h(tp + vec3(0., y, 0.), 2.8, ts, 0., fog(st, tp));
        }
        for band in [0.3f32, 0.9] { ring_h(tp + vec3(0., band, 0.), 2.85, ts, 0., fog(scale(sick(), lum), tp)); }
        for k in 0..cnt(eco, 12) {
            let a = TAU * k as f32 / cnt(eco, 12) as f32;
            draw_line_3d(tp + vec3(a.cos() * 2.8, 0., a.sin() * 2.8), tp + vec3(a.cos() * 2.8, 7.5, a.sin() * 2.8), fog(scale(st, 0.8), tp));
        }
        let level = 2.5 + 1.6 * (0.5 + 0.5 * (t * 0.4 + i as f32 * 2.0).sin()) + f.bass * 0.8;
        ring_h(tp + vec3(0., level, 0.), 2.6, ts, 0., scale(acid(), 0.55 + 0.4 * glow));
        ring_h(tp + vec3(0., 7.5, 0.), 2.8, ts, 0., fog(st, tp));
        for k in 0..6 {
            let a = TAU * k as f32 / 6.0;
            draw_line_3d(tp + vec3(a.cos() * 2.8, 7.5, a.sin() * 2.8), tp + vec3(0., 8.3, 0.), fog(scale(st, 0.9), tp));
        }
    }

    // ---------------------------------------------------------------- pipelines with flowing liquid
    let routes: [&[Vec3]; 4] = [
        &[vec3(4.4, 3.0, 0.), vec3(15., 3.0, 0.), vec3(15., 3.0, -6.2)],
        &[vec3(-4.4, 6.0, 0.), vec3(-17., 6.0, 0.), vec3(-17., 6.0, -3.2)],
        &[vec3(0., 2.0, 4.4), vec3(0., 2.0, 14.), vec3(10.2, 2.0, 14.)],
        &[vec3(15., 7.0, -9.), vec3(15., 10., -9.), vec3(-17., 10., -9.), vec3(-17., 10., -6.), vec3(-17., 7., -6.)],
    ];
    let pipe_n = if eco { 2 } else { 4 };
    for (ri, route) in routes.iter().take(pipe_n).enumerate() {
        for w in route.windows(2) {
            duct(w[0], w[1], 0.32, fog(scale(st, 1.05), w[0]), if eco { 6.0 } else { 3.0 });
            flow(w[0], w[1], t, flow_speed, ri as f32 * 0.23, mixc(acid(), sick(), 0.5 * core));
        }
        // a valve wheel mid-route
        let mid = route[0].lerp(route[1], 0.5) + vec3(0., 0.6, 0.);
        ring_h(mid, 0.4, 8, t * 0.1, scale(sick(), lum));
    }

    // ---------------------------------------------------------------- barrel yard
    let yard = vec3(-8., 0., 16.);
    let n_b = cnt(eco, 10);
    let layout = [
        (0.0f32, 0.75f32, 0.0f32), (1.5, 0.75, 0.0), (3.0, 0.75, 0.0), (4.5, 0.75, 0.0),
        (0.75, 2.25, 0.0), (2.25, 2.25, 0.0), (3.75, 2.25, 0.0), (1.5, 3.75, 0.0),
        (-1.8, 0.62, 2.0), (6.5, 0.62, 1.6),
    ];
    for (i, &(x, y, z)) in layout.iter().take(n_b).enumerate() {
        let c = yard + vec3(x, y, z);
        let tipped = i >= 8;
        let axis = if tipped { vec3(1.0, 0.18, (i as f32 * 1.3).sin() * 0.3) } else { vec3(0., 1., 0.) };
        barrel(c, axis, fog(scale(rust(), lum), c), if eco { 6 } else { 8 });
        if tipped {
            let mouth = c + axis.normalize() * 0.75;
            pool(vec3(mouth.x + 0.5, 0.03, mouth.z), 1.7, t, i as f32, glow, eco);
            for d in 0..cnt(eco, 3) {
                let age = (t * 0.9 + d as f32 / 3.0 + i as f32 * 0.2).fract();
                let p = vec3(mouth.x + 0.4, mouth.y - age * mouth.y, mouth.z);
                draw_line_3d(p, p - vec3(0., 0.2, 0.), scale(acid(), 0.9));
            }
        } else if i % 3 == 0 {
            ring_h(c + vec3(0., 0.78, 0.), 0.45, 8, 0., scale(acid(), 0.4 + 0.5 * glow));
        }
    }

    // ---------------------------------------------------------------- pylons and cables
    let pylons = [vec3(-24., 0., 8.), vec3(24., 0., 8.), vec3(-24., 0., -8.), vec3(24., 0., -8.)];
    let targets = [vec3(-4.2, 8.5, 1.5), vec3(4.2, 8.5, 1.5), vec3(-4.2, 8.5, -1.5), vec3(4.2, 8.5, -1.5)];
    for (i, &p) in pylons.iter().enumerate() {
        let top = p + vec3(0., 11., 0.);
        let pc = fog(st, p);
        draw_line_3d(p, top, pc);
        draw_line_3d(top + vec3(-3., 0., 0.), top + vec3(3., 0., 0.), pc);
        draw_line_3d(top + vec3(-2.2, -2., 0.), top + vec3(2.2, -2., 0.), pc);
        if !eco {
            for s in 0..3 {
                let y0 = s as f32 * 3.6;
                draw_line_3d(p + vec3(-0.9 + s as f32 * 0.25, y0, 0.), p + vec3(0.9 - s as f32 * 0.25, y0 + 3.6, 0.), scale(pc, 0.7));
                draw_line_3d(p + vec3(0.9 - s as f32 * 0.25, y0, 0.), p + vec3(-0.9 + s as f32 * 0.25, y0 + 3.6, 0.), scale(pc, 0.7));
            }
        }
        let tgt = targets[i];
        let end = top + vec3(if p.x < 0. { 3.0 } else { -3.0 }, 0., 0.);
        let mut last = end;
        for s in 1..=8 {
            let q = cable_pt(end, tgt, s as f32 / 8.0, t, 2.2);
            draw_line_3d(last, q, scale(Color::new(0.45, 0.5, 0.4, 1.0), lum));
            last = q;
        }
        // current pulses running down the cable; they brighten with the mids
        for k in 0..2 {
            let u = (t * 0.5 + i as f32 * 0.19 + k as f32 * 0.5).fract();
            spark(cable_pt(end, tgt, u, t, 2.2), 0.18 + 0.12 * f.mid, mixc(sick(), Color::new(1., 1., 0.8, 1.), 0.5));
        }
    }
    // electric arcs on snare hits (lines only, so they can't wash out the screen)
    if f.snare > 0.3 {
        let seed = (t * 30.0).floor();
        let i = (seed as usize) % 4;
        let end = pylons[i] + vec3(0., 11., 0.) + vec3(if pylons[i].x < 0. { 3.0 } else { -3.0 }, 0., 0.);
        let mid = cable_pt(end, targets[i], 0.5, t, 2.2);
        let ground = vec3(mid.x + (seed * 1.7).sin() * 3.0, 0.1, mid.z + (seed * 2.3).cos() * 3.0);
        bolt(mid, ground, seed, 0.9, Color::new(0.8, 1.0, 0.7, 1.0));
        if !eco { bolt(mid, ground, seed + 5.0, 0.6, Color::new(0.4, 0.9, 0.3, 1.0)); }
        spark(ground, 0.5, Color::new(0.9, 1.0, 0.8, 1.0));
    }

    // ---------------------------------------------------------------- searchlight and hazard signs
    let tower = vec3(24., 0., 24.);
    let lamp = tower + vec3(0., 9., 0.);
    draw_line_3d(tower, lamp, st);
    draw_line_3d(tower + vec3(1., 0., 0.), lamp, scale(st, 0.7));
    draw_line_3d(tower + vec3(-1., 0., 0.), lamp, scale(st, 0.7));
    let sa = t * 0.35;
    let g = vec3(sa.cos() * (12. + 4. * (t * 0.2).sin()), 0.05, sa.sin() * (12. + 4. * (t * 0.2).sin()));
    let bc = scale(sick(), 0.5 + 0.3 * f.mid);
    for k in 0..3 {
        let a = TAU * k as f32 / 3.0;
        draw_line_3d(lamp, g + vec3(a.cos() * 2.2, 0., a.sin() * 2.2), bc);
    }
    ring_h(g, 2.2, 12, 0., bc);
    spark(lamp, 0.5, sick());
    trefoil(vec3(-12., 4.2, 9.), 1.2, scale(sick(), lum));
    draw_line_3d(vec3(-12., 0., 9.), vec3(-12., 3.0, 9.), st);
    if !eco {
        trefoil(vec3(19.5, 3.4, -4.5), 0.9, scale(sick(), lum));
        draw_line_3d(vec3(19.5, 0., -4.5), vec3(19.5, 2.5, -4.5), st);
    }

    // ---------------------------------------------------------------- vapour, smoke and sparks
    let wind = (t * 0.3).sin() * 1.5 + 0.7;
    let vents: [Vec3; 12] = [
        tanks[0] + vec3(0., 8.3, 0.), tanks[1] + vec3(0., 8.3, 0.), tanks[2] + vec3(0., 8.3, 0.),
        vec3(-9., 0.3, -17.), vec3(5., 0.3, -18.), vec3(22., 0.3, 3.), vec3(-8., 0.3, 8.),
        vec3(-6., 0.8, 18.), vec3(-1.5, 0.8, 18.), vec3(6., 1., 12.), vec3(-14., 0.3, -2.), vec3(10., 0.3, 6.),
    ];
    for i in 0..cnt(eco, 12) {
        let v0 = vents[i];
        let wp = h01(i as u32, 2) * TAU;
        let mut last = v0;
        for j in 1..=7 {
            let h = j as f32 * 1.25;
            let w = 0.25 + h * 0.08;
            let p = v0 + vec3((h * 0.9 + t * 0.6 + wp).sin() * w + wind * h * 0.12, h, (h * 0.8 + t * 0.5 + wp).cos() * w);
            draw_line_3d(last, p, scale(Color::new(0.55, 0.85, 0.2, 1.0), 0.55 * (1.0 - j as f32 / 8.5)));
            last = p;
        }
    }
    // main plume rising from the reactor dome
    for lane in 0..cnt(eco, 3) {
        let mut last = vec3(0., 11.3, 0.);
        for j in 1..=9 {
            let h = j as f32 * 1.6;
            let spread = 0.3 + h * 0.18;
            let off = (lane as f32 - 1.0) * spread;
            let p = vec3(off + (h * 0.5 + t * 0.4).sin() * 0.9 + wind * h * 0.3, 11.3 + h, (h * 0.4 + t * 0.35 + lane as f32).cos() * 0.9);
            draw_line_3d(last, p, scale(Color::new(0.38, 0.5, 0.22, 1.0), 0.8 * (1.0 - j as f32 / 10.5)));
            last = p;
        }
    }
    // sparks: thrown up from the cable junctions at the reactor shoulder, more with treble
    for i in 0..cnt(eco, 14) {
        let u = i as u32 + 40;
        let age = (t * 0.9 + h01(u, 1)).fract();
        let src = targets[i % 4];
        let ang = h01(u, 2) * TAU;
        let sp = 2.0 + 3.0 * h01(u, 3);
        let p = src + vec3(ang.cos() * sp * age, 5.0 * age - 7.0 * age * age, ang.sin() * sp * age);
        spark(p, (0.08 + f.treble * 0.12) * (1.0 - age), mixc(Color::new(1., 0.95, 0.4, 1.), Color::new(0.8, 0.35, 0.05, 1.), age));
    }
    // radiation flecks: brief flickers near the ground, busier with treble
    for i in 0..cnt(eco, 24) {
        let u = i as u32 + 200;
        if h01(u, (t * 8.0 + i as f32).floor() as u32) > 0.78 - 0.2 * f.treble {
            spark(vec3((h01(u, 1) - 0.5) * 60.0, 0.1 + h01(u, 2) * 3.0, (h01(u, 3) - 0.5) * 60.0), 0.08 + f.treble * 0.1, sick());
        }
    }

    // ---------------------------------------------------------------- skyline of cooling towers
    for (i, &(x, z)) in [(-42.0f32, -62.0f32), (0.0, -72.0), (42.0, -62.0)].iter().enumerate() {
        if eco && i != 1 { continue; }
        let ts = if eco { 8 } else { 10 };
        let r_at = |y: f32| 5.0 + 3.5 * ((y - 14.0) / 14.0).powi(2);
        let base = vec3(x, 0., z);
        let tc = fog(Color::new(0.20, 0.26, 0.20, 1.0), base);
        let mut prev: Vec<f32> = Vec::new();
        let _ = &mut prev;
        for k in 0..=7 {
            let y = k as f32 * 4.0;
            ring_h(base + vec3(0., y, 0.), r_at(y), ts, 0., tc);
            if k > 0 {
                let y0 = y - 4.0;
                for s in (0..ts).step_by(if eco { 2 } else { 2 }) {
                    let a = TAU * s as f32 / ts as f32;
                    draw_line_3d(base + vec3(a.cos() * r_at(y0), y0, a.sin() * r_at(y0)), base + vec3(a.cos() * r_at(y), y, a.sin() * r_at(y)), scale(tc, 0.8));
                }
            }
        }
        let mut last = base + vec3(0., 28., 0.);
        for j in 1..=6 {
            let h = j as f32 * 2.2;
            let p = base + vec3((h * 0.4 + t * 0.3 + i as f32).sin() * 1.4 + wind * h * 0.25, 28. + h, 0.);
            draw_line_3d(last, p, scale(Color::new(0.35, 0.42, 0.3, 1.0), 0.6 * (1.0 - j as f32 / 7.5)));
            last = p;
        }
        if ((t * 0.5 + i as f32 * 0.3) % 1.0) < 0.5 { spark(base + vec3(0., 28.5, 0.), 0.4, Color::new(1., 0.15, 0.1, 1.)); }
    }
}
}
use audio::{AudioInput, Levels, BANDS};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use macroquad::prelude::*;

const FIREFLY_COUNT: usize = 12;
const MAX_TRAIL: usize = 8;
const THEMES: [&str; 11] = ["Toxic","Synthwave","Beat Sync","Rainbow Crystal","Iron Man","Snow","Matrix","Fire","Ocean","Firefly Forest","Cyber Firefly"];

fn palette(theme:usize,t:f32)->Color{
 match theme {0=>GREEN,1=>PINK,2=>SKYBLUE,3=>Color::new((t*0.7).sin().abs(),(t*0.5+2.).sin().abs(),(t*0.4+4.).sin().abs(),1.),4=>RED,5=>WHITE,6=>GREEN,7=>ORANGE,8=>BLUE,9=>GOLD,_=>Color::new(0.15,0.94,0.22,1.)}
}

// Cyber Firefly V3: audio-reactive swarm with layered glow, wings and long trails.
struct Firefly {
    pos: Vec3, vel: Vec3, phase: f32, size: f32,
    trail: VecDeque<Vec3>,
}
impl Firefly {
    fn new() -> Self {
        Self {
            pos: vec3(rand::gen_range(-19.,19.),rand::gen_range(3.,16.),rand::gen_range(-19.,19.)),
            vel: vec3(rand::gen_range(-3.,3.),rand::gen_range(-1.,1.),rand::gen_range(-3.,3.)),
            phase: rand::gen_range(0.,std::f32::consts::TAU),
            size: rand::gen_range(0.17,0.32),
            trail: VecDeque::with_capacity(MAX_TRAIL),
        }
    }
    fn update(&mut self, dt:f32, time:f32, bass:f32, treble:f32, center:Vec3, separation:Vec3) {
        // Independently phased, multi-frequency wander; soft steering avoids
        // artificial synchronised orbits and sudden boundary clipping.
        let t=time*(0.36+bass*0.24)+self.phase*1.7;
        let id=self.phase;
        let destination=vec3(
            (t*(0.48+id.sin().abs()*0.26)).sin()*23.0
                +(t*0.19+id*2.3).cos()*9.0,
            9.0+(t*(0.68+id.cos().abs()*0.25)).sin()*5.5
                +(t*0.27+id).cos()*3.0,
            (t*(0.37+id.cos().abs()*0.29)+id).cos()*23.0
                +(t*0.23+id*2.1).sin()*9.0
        );
        let desired=(destination-self.pos)*0.24
            +(center-self.pos)*0.005
            +separation*2.0;
        self.vel+=(desired-self.vel)*dt*1.15;
        let max_speed=5.8+bass*4.5;
        let speed=self.vel.length();
        if speed>max_speed {self.vel*=max_speed/speed;}
        self.pos+=self.vel*dt;
        // Spring-like recovery at the far edge of the landscape.
        if self.pos.x.abs()>38. {self.vel.x-=self.pos.x.signum()*dt*9.;}
        if self.pos.z.abs()>38. {self.vel.z-=self.pos.z.signum()*dt*9.;}
        if self.pos.y<2.0 {self.vel.y+=dt*8.;}
        if self.pos.y>23.0 {self.vel.y-=dt*8.;}
        self.trail.push_back(self.pos);
        while self.trail.len()>MAX_TRAIL {self.trail.pop_front();}
        let _=treble;
    }
    fn shock(&mut self, strength:f32) {
        let outward=vec3(self.pos.x,self.pos.y-7.,self.pos.z);
        if outward.length()>0.01 {self.vel+=outward.normalize()*strength;}
    }
    fn draw(&self,time:f32,treble:f32,bass:f32) {
        let glow=(time*3.1+self.phase).sin()*0.5+0.5;
        let cyan=Color::new(0.22+glow*0.30,0.90+glow*0.10,0.025,1.);
        let violet=Color::new(0.03,0.48+glow*0.48,0.05,1.);
        let radius=self.size*(0.82+bass*0.65+0.36*glow);
        let n=self.trail.len();
        for i in 1..n {
            let f=i as f32/n as f32;
            let p=self.trail[i-1];let q=self.trail[i];
            draw_line_3d(p,q,Color::new(0.06+0.18*f,0.35+0.6*f,0.03,1.));
            
        }
        // Layered additive-looking halo: multiple translucent shells and bright core.
        
        draw_sphere(self.pos,radius*1.1,None,cyan);
        draw_line_3d(self.pos+vec3(0.,radius,0.),self.pos-vec3(0.,radius,0.),WHITE);
        let flap=(time*(13.+treble*46.)+self.phase).sin();
        let span=radius*(3.5+flap.abs()*2.5);
        // Four triangular wings, moving with high-frequency energy.
        for side in [-1.0_f32,1.0] {
            for fore in [-1.0_f32,1.0] {
                let root=self.pos+vec3(side*radius*0.3,0.,0.);
                let tip=self.pos+vec3(side*span,fore*radius*(1.3+flap*1.5),fore*span*0.75);
                let back=self.pos+vec3(side*span*0.35,-fore*radius*0.7,fore*span*0.55);
                draw_line_3d(root,tip,cyan);
                draw_line_3d(tip,back,violet);
                draw_line_3d(back,root,cyan);
            }
        }
    }
}
// Low-cost animated wireframe ocean: no mesh allocations or textures.
fn ocean_height(x:f32,z:f32,t:f32,bass:f32,mid:f32)->f32 {
    let swell=(x*0.18+t*0.95).sin()*(1.1+bass*3.2);
    let cross=(z*0.22-t*0.75).cos()*(0.65+mid*2.0);
    let ripples=(x*0.42+z*0.35+t*1.9).sin()*0.28;
    -1.0+swell+cross+ripples
}
// Low-poly swaying grass tufts; deterministic positions, no particle allocations.
// Earth-tone terrain, scattered faceted boulders and a moving stream at the side.
fn forest_river_x(z:f32)->f32 { 18.0+(z*0.105).sin()*2.8+(z*0.045+1.3).sin()*1.0 }
fn forest_ground(x:f32,z:f32)->f32 {
 -0.36+0.25*(x*0.21).sin()+0.19*(z*0.27).cos()+0.12*(x*0.36+z*0.14).sin()
}
fn draw_firefly_landscape(t:f32,bass:f32,eco:bool) {
 // Earthy woodland ground with varied green moss and river-carved banks.
 let step=if eco {4.2_f32} else {2.7_f32};
 let n=if eco {10} else {17};
 for ix in -n..n {
  let x=ix as f32*step;
  for iz in -n..n {
   let z=iz as f32*step;
   if (x-forest_river_x(z)).abs()<4.5 {continue;}
   let y=forest_ground(x,z);
   let tint=(x*0.43+z*0.21).sin()*0.5+0.5;
   let soil=Color::new(0.20+tint*0.10,0.23+tint*0.12,0.085,1.);
   let moss=Color::new(0.13,0.31+tint*0.14,0.08,1.);
   draw_line_3d(vec3(x,y,z),vec3(x+step,forest_ground(x+step,z),z),soil);
   draw_line_3d(vec3(x,y,z),vec3(x,forest_ground(x,z+step),z+step),moss);
  }
 }
 // Irregular low-poly boulders (not box-shaped rocks).
 let rocks=if eco {17} else {36};
 for i in 0..rocks {
  let f=i as f32;
  let z=(f*19.31).sin()*27.;
  let x=(f*8.37).cos()*22.;
  if (x-forest_river_x(z)).abs()<4.7 {continue;}
  let r=0.55+(f*2.83).sin().abs()*1.10;
  let centre=vec3(x,forest_ground(x,z)+r*0.45,z);
  let base=if i%3==0 {Color::new(0.36,0.32,0.25,1.)}
      else {Color::new(0.28,0.31,0.28,1.)};
  let top=Color::new(0.27,0.39,0.21,1.);
  let segments=if eco {5} else {7};
  let mut previous_low=Vec3::ZERO;
  let mut previous_mid=Vec3::ZERO;
  for k in 0..=segments {
   let angle=std::f32::consts::TAU*k as f32/segments as f32;
   let jag=0.85+0.16*(f*5.7+k as f32*2.3).sin();
   let low=centre+vec3(angle.cos()*r*jag,-r*0.43,angle.sin()*r*0.8*jag);
   let mid=centre+vec3(angle.cos()*r*0.82*jag,r*0.22,angle.sin()*r*0.7*jag);
   let peak=centre+vec3((f*1.7).sin()*r*0.17,r*0.68,(f*2.3).cos()*r*0.17);
   if k>0 {
    draw_line_3d(previous_low,low,base);
    draw_line_3d(previous_mid,mid,base);
    draw_line_3d(previous_mid,peak,top);
    if !eco {draw_line_3d(previous_low,mid,base);}
   }
   previous_low=low;previous_mid=mid;
  }
 }
 // Layered, curved river surface with dark water and moving light reflections.
 let slices=if eco {46} else {84};
 let lanes=if eco {5} else {9};
 for j in 0..slices {
  let z=-46.0+j as f32*92.0/slices as f32;
  let z2=-46.0+(j+1) as f32*92.0/slices as f32;
  let c=forest_river_x(z);
  let c2=forest_river_x(z2);
  for k in 0..lanes {
   let f=k as f32/(lanes-1) as f32;
   let x=c+(f-0.5)*8.0;
   let x2=c2+(f-0.5)*8.0;
   let ripple=(z*0.65-t*(1.8+bass*2.4)+f*8.0).sin()*0.07;
   let ripple2=(z2*0.65-t*(1.8+bass*2.4)+f*8.0).sin()*0.07;
   let y=-0.52+ripple;
   let shade=0.5+0.5*(z*0.24+f*9.0).sin();
   let water=Color::new(0.025+shade*0.05,0.19+shade*0.19,0.22+shade*0.19,1.);
   draw_line_3d(vec3(x,y,z),vec3(x2,-0.52+ripple2,z2),water);
   if j%3==0 && k%2==0 {
    let glint=Color::new(0.12+shade*0.17,0.48+shade*0.27,0.41+shade*0.25,1.);
    draw_line_3d(vec3(x,y+0.018,z),vec3(x+0.55,y+0.018,z+0.15),glint);
   }
  }
  for side in [-1.0_f32,1.0] {
   let edge=c+side*4.1;
   let edge2=c2+side*4.1;
   draw_line_3d(vec3(edge,-0.35,z),vec3(edge2,-0.35,z2),Color::new(0.42,0.34,0.22,1.));
   draw_line_3d(vec3(edge,-0.35,z),vec3(edge+side*1.5,0.35,z),Color::new(0.18,0.36,0.12,1.));
  }
 }
 // Downstream foam flecks; deterministic looping paths and light shimmer.
 for i in 0..(if eco {24} else {66}) {
  let f=i as f32;
  let z=(f*13.37+t*(2.0+bass*3.2)).rem_euclid(92.0)-46.0;
  let x=forest_river_x(z)+(f*8.3).sin()*3.4;
  let y=-0.43+(t*2.7+f).sin()*0.045;
  let c=if i%3==0 {Color::new(0.48,0.81,0.74,1.)}
       else {Color::new(0.20,0.55,0.55,1.)};
  draw_line_3d(vec3(x,y,z),vec3(x+0.24,y,z+0.52),c);
 }
}
// Low-cost fern fronds clustered beside the river and forest boulders.
fn draw_firefly_ferns(t:f32,bass:f32,treble:f32,eco:bool) {
 let count=if eco {22} else {49};
 let leaf=if eco {4} else {7};
 for i in 0..count {
  let f=i as f32;
  let z=(f*9.713).sin()*41.0;
  let side=if i%2==0 {-1.0_f32} else {1.0};
  let x=forest_river_x(z)+side*(5.7+(f*3.19).sin().abs()*16.0);
  let ground=forest_ground(x,z);
  let height=0.7+(f*1.83).sin().abs()*1.25;
  let wind=(t*1.2+f*2.0).sin()*(0.11+bass*0.35+treble*0.12);
  let base=vec3(x,ground,z);
  let green=if i%3==0 {Color::new(0.20,0.48,0.14,1.)}
            else if i%3==1 {Color::new(0.29,0.49,0.17,1.)}
            else {Color::new(0.12,0.37,0.15,1.)};
  for arm in 0..5 {
   let angle=std::f32::consts::TAU*arm as f32/5.0+f;
   let dx=angle.cos();let dz=angle.sin();
   let reach=0.9+(arm as f32*1.7+f).sin().abs()*0.65;
   let mut previous=base;
   for j in 1..=leaf {
    let u=j as f32/leaf as f32;
    let center=base+vec3(dx*reach*u+wind*u*u,height*(u-u*u*0.5),dz*reach*u);
    draw_line_3d(previous,center,green);
    let width=(1.0-u)*0.32;
    let normal=vec3(-dz,0.,dx);
    for direction in [-1.0_f32,1.0] {
     let tip=center+normal*width*direction+vec3(0.,-width*0.23,0.);
     draw_line_3d(center,tip,green);
    }
    previous=center;
   }
  }
 }
}
// V39: mossy fallen logs, miniature river cascades and glowing mushroom colonies.
// Deterministic placement avoids allocations and preserves ECO performance.
// V40: ancient trees, exposed roots, hanging vines and drifting luminous pollen.
fn draw_firefly_ancient_grove(t:f32,bass:f32,treble:f32,eco:bool) {
 let bark=Color::new(0.29,0.21,0.14,1.);
 let shadow=Color::new(0.17,0.17,0.12,1.);
 let moss=Color::new(0.20,0.43,0.17,1.);
 let canopy=Color::new(0.13,0.37,0.20,1.);
 // Keep roots and trunks clear of the river's central channel.
 for i in 0..(if eco {7} else {13}) {
  let f=i as f32;
  let z=-39.0+(f*17.37).rem_euclid(78.0);
  let side=if i%2==0 {-1.0_f32} else {1.0};
  let x=forest_river_x(z)+side*(10.0+(f*3.71).sin().abs()*17.0);
  let y=forest_ground(x,z);
  let height=10.0+(f*2.83).sin().abs()*9.0;
  let wind=(t*0.57+f*1.71).sin()*(0.17+bass*0.36);
  let base=vec3(x,y,z);
  let middle=base+vec3(wind*0.30,height*0.54,0.35);
  let crown=base+vec3(wind,height,0.7);
  for offset in [-0.72_f32,0.0,0.72] {
   draw_line_3d(base+vec3(offset,0.,0.),middle+vec3(offset*0.57,0.,0.),if offset==0. {bark} else {shadow});
   draw_line_3d(middle+vec3(offset*0.57,0.,0.),crown+vec3(offset*0.35,0.,0.),bark);
  }
  // Large exposed roots, radiating outward and wrapping around stones.
  for r in 0..(if eco {5} else {8}) {
   let rf=r as f32;
   let a=std::f32::consts::TAU*rf/(if eco {5.0} else {8.0})+f;
   let reach=2.8+(rf*1.9+f).sin().abs()*2.9;
   let knee=base+vec3(a.cos()*reach*0.43,0.55,a.sin()*reach*0.43);
   let endx=x+a.cos()*reach;
   let endz=z+a.sin()*reach;
   let end=vec3(endx,forest_ground(endx,endz)+0.08,endz);
   draw_line_3d(base+vec3(a.cos()*0.65,1.3,a.sin()*0.65),knee,bark);
   draw_line_3d(knee,end,shadow);
   if r%2==0 {draw_line_3d(knee+vec3(0.,0.10,0.),end+vec3(0.,0.08,0.),moss);}
  }
  // Long horizontal limbs, twigs, and hanging vines.
  for b in 0..(if eco {4} else {7}) {
   let bf=b as f32;
   let a=bf*2.39996+f;
   let level=height*(0.56+bf*0.045);
   let start=base+vec3(wind*0.5,level,0.3);
   let reach=3.2+(bf*1.77+f).sin().abs()*3.8;
   let end=start+vec3(a.cos()*reach,1.2+(bf*1.3).sin(),a.sin()*reach);
   draw_line_3d(start,end,bark);
   draw_line_3d(start.lerp(end,0.63),end+vec3(a.sin()*0.7,1.4,a.cos()*0.7),moss);
   let vine_count=if eco {1} else {3};
   for v in 0..vine_count {
    let u=0.55+v as f32*0.16;
    let anchor=start.lerp(end,u);
    let length=1.8+(f+bf+v as f32).sin().abs()*3.6;
    let segments=if eco {3} else {6};
    let mut prev=anchor;
    for seg in 1..=segments {
     let q=seg as f32/segments as f32;
     let sway=(t*0.8+f+bf+q*3.0).sin()*(0.18+treble*0.30)*q;
     let next=anchor+vec3(sway,-length*q,(q*4.0+f).sin()*0.15*q);
     draw_line_3d(prev,next,if v%2==0 {moss} else {canopy});
     prev=next;
    }
   }
  }
 }
 // Firefly-lit pollen motes drift independently of the camera and pulse with treble.
 for i in 0..(if eco {34} else {95}) {
  let f=i as f32;
  let x=((f*17.19).sin()*137.0+t*(0.09+treble*0.22)).rem_euclid(85.0)-42.5;
  let z=((f*31.7).cos()*193.0+t*0.14).rem_euclid(88.0)-44.0;
  let y=0.7+(f*7.11+t*(0.22+treble*0.10)).rem_euclid(14.0);
  let size=0.035+(f*2.11).sin().abs()*0.10+treble*0.04;
  let glint=0.65+0.35*(t*1.9+f).sin().abs();
  let c=if i%3==0 {Color::new(0.80*glint,0.93*glint,0.32,1.)}
   else {Color::new(0.30*glint,0.87*glint,0.65,1.)};
  let p=vec3(x,y,z);
  draw_line_3d(p+vec3(-size,0.,0.),p+vec3(size,0.,0.),c);
  if !eco && i%4==0 {draw_line_3d(p+vec3(0.,-size,0.),p+vec3(0.,size,0.),c);}
 }
}
fn draw_firefly_stream_details(t:f32,bass:f32,treble:f32,eco:bool) {
 let wood=Color::new(0.37,0.23,0.12,1.);
 let bark=Color::new(0.20,0.15,0.10,1.);
 let moss=Color::new(0.24,0.48,0.19,1.);
 let cyan=Color::new(0.31,0.83,0.89,1.);
 // Two fallen trunks, crossing the water from bank to bank.
 for i in 0..2 {
  let f=i as f32;
  let z=if i==0 {-14.0_f32} else {23.0_f32};
  let cx=forest_river_x(z);
  let a=vec3(cx-6.8,0.40,z-1.0);
  let b=vec3(cx+6.8,0.65,z+1.3);
  for k in 0..(if eco {3} else {5}) {
   let offset=(k as f32-2.0)*0.11;
   draw_line_3d(a+vec3(0.,offset,0.),b+vec3(0.,offset,0.),if k%2==0 {wood} else {bark});
  }
  // Broken branches and patches of moss on the logs.
  for j in 0..4 {
   let u=(j as f32+0.6)/5.0;
   let p=a.lerp(b,u);
   draw_line_3d(p,p+vec3((f+j as f32).sin()*0.45,0.45+j as f32*0.12,0.25),bark);
   draw_line_3d(p+vec3(-0.30,0.07,0.),p+vec3(0.35,0.09,0.),moss);
  }
 }
 // Cascading ripples and bright falling-water strands across two low ledges.
 for cascade in 0..2 {
  let z=if cascade==0 {-28.0_f32} else {31.0_f32};
  let cx=forest_river_x(z);
  let n=if eco {10} else {24};
  for i in 0..n {
   let u=i as f32/(n-1) as f32;
   let x=cx+(u-0.5)*7.5;
   let phase=(t*(1.4+bass*1.6)+u*4.7).fract();
   let y=0.10-phase*0.73;
   let zshift=(phase*0.6)+(u*19.0+t*2.2).sin()*0.08;
   let pale=if i%3==0 {Color::new(0.70,0.94,0.96,1.)} else {cyan};
   draw_line_3d(vec3(x,0.12,z),vec3(x,y,z+zshift),pale);
   draw_line_3d(vec3(x,-0.57,z+0.4),vec3(x+0.22,-0.57,z+0.8+bass*0.45),pale);
  }
 }
 // Bioluminescent mushroom clusters grow on the shaded riverbanks.
 let clusters=if eco {11} else {25};
 for i in 0..clusters {
  let f=i as f32;
  let z=(f*11.37).sin()*40.0;
  let side=if i%2==0 {-1.0_f32} else {1.0};
  let x=forest_river_x(z)+side*(5.0+(f*2.11).sin().abs()*11.0);
  let y=forest_ground(x,z);
  let glow=0.75+0.25*(t*2.1+f*1.8).sin()+treble*0.18;
  let cap=if i%3==0 {Color::new(0.20*glow,0.85*glow,0.92*glow,1.)}
          else if i%3==1 {Color::new(0.60*glow,0.38*glow,0.93*glow,1.)}
          else {Color::new(0.32*glow,0.95*glow,0.50*glow,1.)};
  for j in 0..3 {
   let jf=j as f32;
   let bx=x+(jf-1.0)*0.42;
   let bz=z+(jf*2.7+f).sin()*0.40;
   let h=0.25+0.15*(f+jf).sin().abs()+bass*0.06;
   let stem=vec3(bx,y,bz);
   let top=stem+vec3(0.,h,0.);
   draw_line_3d(stem,top,Color::new(0.54,0.62,0.53,1.));
   draw_line_3d(top+vec3(-0.23,0.,0.),top+vec3(0.23,0.,0.),cap);
   draw_line_3d(top+vec3(-0.23,0.,0.),top+vec3(0.,0.15,0.),cap);
   draw_line_3d(top+vec3(0.,0.15,0.),top+vec3(0.23,0.,0.),cap);
  }
 }
}
fn draw_firefly_grass(t:f32,bass:f32,treble:f32,eco:bool) {
 let patches=if eco {12} else {25};
 let blades=if eco {5} else {9};
 let dark=Color::new(0.06,0.23,0.10,1.0);
 let light=Color::new(0.20,0.55+treble*0.18,0.16,1.0);
 for p in 0..patches {
  let seed=p as f32;
  let x=(seed*17.13).sin()*20.0;
  let z=(seed*31.71).cos()*20.0;
  draw_cube(vec3(x,0.0,z),vec3(2.0,0.12,2.0),None,dark);
  for b in 0..blades {
   let f=b as f32;
   let angle=f*2.399+seed;
   let radius=0.2+(f*5.31).sin().abs()*0.8;
   let root=vec3(x+angle.cos()*radius,0.1,z+angle.sin()*radius);
   let height=0.8+(seed*3.4+f*1.6).sin().abs()*1.5;
   let sway=(t*(1.1+treble*1.7)+seed+f*0.6).sin()*(0.28+bass*0.6);
   let tip=root+vec3(sway,height,sway*0.5);
   let middle=root+vec3(sway*0.25,height*0.55,sway*0.2);
   draw_line_3d(root,middle,dark);
   draw_line_3d(middle,tip,light);
  }
 }
}
// Layered, irregular alpine ridges and gently drifting snow.
// Deterministic terrain and flakes: no per-frame allocations.
fn alpine_height(x:f32,z:f32,side:f32)->f32 {
 let u=z*0.11+side*1.6;
 let ridge=(u*0.69).sin().abs()*8.5+(u*1.17+0.8).sin().abs()*5.2;
 let fractured=(u*2.4+x*0.04).sin().abs()*3.2+(u*4.8).cos().abs()*1.7;
 let slope=((x.abs()-11.0)/18.0).clamp(0.0,1.0);
 -0.7+slope*(5.5+ridge+fractured)
}
fn draw_blizzard(t:f32,bass:f32,treble:f32,eco:bool) {
 let snow=Color::new(0.78,0.87,0.94,1.);
 let grey=Color::new(0.37,0.40,0.43,1.);
 let brown=Color::new(0.43,0.34,0.29,1.);
 // Frozen valley with subtle ice cracks and reflected sunlight.
 let spacing=if eco {4.5_f32} else {2.8_f32};
 let n=if eco {10} else {17};
 for iz in -n..n {
  let z=iz as f32*spacing;
  for ix in -n..n {
   let x=ix as f32*spacing;
   let frozen=x.abs()<12.5+(z*0.08).sin()*1.4;
   let y=if frozen {-0.77} else {-0.48+0.09*(x*0.3).sin()};
   let c=if frozen {
     let shimmer=0.5+0.5*(x*0.24+z*0.19).sin();
     Color::new(0.24+shimmer*0.16,0.42+shimmer*0.16,0.55+shimmer*0.17,1.)
   } else {snow};
   draw_line_3d(vec3(x,y,z),vec3(x+spacing,y,z),c);
   if !eco || iz%2==0 {
    draw_line_3d(vec3(x,y,z),vec3(x,y,z+spacing),c);
   }
  }
 }
 // Jagged, multi-ridge mountains: grey/brown rock faces and snowcap seams.
 let slices=if eco {34} else {62};
 for side in [-1.0_f32,1.0] {
  for layer in 0..3 {
   let inset=layer as f32*5.5;
   let mut last_top:Option<Vec3>=None;
   let mut last_foot:Option<Vec3>=None;
   let mut last_mid:Option<Vec3>=None;
   for j in 0..=slices {
    let z=-52.0+104.0*j as f32/slices as f32;
    let x=side*(18.0+inset+(z*0.075+layer as f32).sin()*2.1);
    let peak=alpine_height(x,z+layer as f32*12.0,side)*(1.0-layer as f32*0.12);
    let top=vec3(x,peak,z);
    let foot=vec3(side*(10.0+inset),-0.55,z);
    let mid=foot.lerp(top,0.58+(z*0.17).sin()*0.07);
    if let (Some(pt),Some(pf),Some(pm))=(last_top,last_foot,last_mid) {
     let rock=if (j+layer)%3==0 {brown} else {grey};
     draw_line_3d(pt,top,if peak>12.0 {snow} else {rock});
     draw_line_3d(pf,foot,snow);
     draw_line_3d(pm,mid,rock);
     if j%2==0 {
      draw_line_3d(foot,mid,rock);
      draw_line_3d(mid,top,snow);
      if !eco {draw_line_3d(pm,top,rock);}
     }
    }
    last_top=Some(top);last_foot=Some(foot);last_mid=Some(mid);
   }
  }
 }
 // Frozen river/lake: long glints mimic sky and mountain reflections.
 let reflections=if eco {42} else {110};
 for i in 0..reflections {
  let f=i as f32;
  let z=(f*19.27).sin()*39.0;
  let x=(f*13.11).cos()*11.0;
  let light=(0.5+0.5*(z*0.16+t*0.20+f).sin()).powf(2.0);
  let color=Color::new(0.45+light*0.40,0.60+light*0.32,0.72+light*0.22,1.);
  let length=0.3+light*2.2;
  draw_line_3d(vec3(x,-0.745,z),vec3(x+length,-0.745,z+0.1),color);
 }
 // Branching hairline cracks in the ice, stable as camera moves.
 for i in 0..(if eco {12} else {26}) {
  let f=i as f32;
  let z=(f*8.11).sin()*35.0;
  let x=(f*4.71).cos()*9.5;
  let a=vec3(x,-0.74,z);
  let b=a+vec3(0.8+(f*2.3).sin(),0.,1.3+(f*1.1).cos());
  let c=b+vec3(1.2,0.,0.4);
  let crack=Color::new(0.18,0.34,0.45,1.);
  draw_line_3d(a,b,crack);
  draw_line_3d(b,c,crack);
  if !eco {draw_line_3d(b,b+vec3(-0.8,0.,0.9),crack);}
 }
 // Gently drifting flakes, consistent density and no large harsh streaks.
 let count=if eco {170} else {350};
 for i in 0..count {
  let f=i as f32;
  let r=((f*12.9898).sin()*43758.5453).fract().abs();
  let r2=((f*31.71+4.2).sin()*17342.123).fract().abs();
  let depth=r2*92.0-46.0;
  let speed=0.75+r*1.3+bass*0.30;
  let wind=1.25+treble*1.5+bass*0.45;
  let x=(r*96.0+t*wind+(t*0.55+f).sin()*0.7).rem_euclid(96.0)-48.0;
  let y=(r2*35.0-t*speed).rem_euclid(35.0)-1.0;
  let pos=vec3(x,y,depth);
  let radius=0.045+r*0.12;
  let c=if r>0.58 {Color::new(0.96,0.97,1.0,1.)}
        else {Color::new(0.68,0.78,0.87,1.)};
  draw_line_3d(pos+vec3(-radius,0.,0.),pos+vec3(radius,0.,0.),c);
  draw_line_3d(pos+vec3(0.,-radius,0.),pos+vec3(0.,radius,0.),c);
  if !eco && r>0.63 {
   draw_line_3d(pos+vec3(-radius*0.6,-radius*0.6,0.),
                pos+vec3(radius*0.6,radius*0.6,0.),c);
   draw_line_3d(pos+vec3(-radius*0.6,radius*0.6,0.),
                pos+vec3(radius*0.6,-radius*0.6,0.),c);
  }
 }
}
fn draw_ocean_splash(x:f32,z:f32,y:f32,age:f32,eco:bool) {
 let radius=0.15+age*2.5;
 let fade=(1.0-age/0.75).clamp(0.0,1.0);
 let segments=if eco {8} else {14};
 let c=Color::new(0.28*fade,0.65*fade,0.89*fade,1.0);
 for k in 0..segments {
  let a=std::f32::consts::TAU*k as f32/segments as f32;
  let b=std::f32::consts::TAU*(k+1) as f32/segments as f32;
  draw_line_3d(vec3(x+radius*a.cos(),y+0.12,z+radius*a.sin()),
               vec3(x+radius*b.cos(),y+0.12,z+radius*b.sin()),c);
 }
}
fn draw_snow_moon(t:f32,eco:bool) {
 // Crescent-like half-moon disk: half illuminated, half dark, on the far sky.
 let center=vec3(-35.0,38.0,-65.0);
 let r=if eco {5.0} else {6.0};
 let segments=if eco {16} else {30};
 let light=Color::new(0.89,0.92,0.80,1.0);
 let dark=Color::new(0.11,0.17,0.23,1.0);
 for k in 0..segments {
  let a=std::f32::consts::TAU*k as f32/segments as f32;
  let b=std::f32::consts::TAU*(k+1) as f32/segments as f32;
  let p=vec3(center.x+r*a.cos(),center.y+r*a.sin(),center.z);
  let q=vec3(center.x+r*b.cos(),center.y+r*b.sin(),center.z);
  draw_line_3d(p,q,light);
  // Horizontal stripes make the lit left hemisphere visibly filled.
  let y=center.y-r+(2.0*r)*(k as f32+0.5)/segments as f32;
  let h=y-center.y;
  let half=(r*r-h*h).max(0.).sqrt();
  draw_line_3d(vec3(center.x-half,y,center.z),vec3(center.x,y,center.z),light);
  draw_line_3d(vec3(center.x,y,center.z),vec3(center.x+half,y,center.z),dark);
 }
 let _=t;
}

// Low-cost cloud banks and moving moonlit wisps over the alpine valley.
fn draw_snow_clouds(t:f32,eco:bool) {
 let banks=if eco {9} else {17};
 for i in 0..banks {
  let f=i as f32;
  let z=-55.0+(f*13.71).rem_euclid(105.0);
  let x=(f*8.33).sin()*43.0+(t*0.35+f*0.2).sin()*3.0;
  let y=25.0+(f*2.31).sin()*5.0;
  let grey=Color::new(0.29,0.36,0.45,1.);
  let silver=Color::new(0.57,0.64,0.72,1.);
  // Thin overlapping contour bands avoid heavy transparent meshes.
  for band in 0..(if eco {3} else {5}) {
   let b=band as f32;
   let width=4.5+b*2.2+(f*1.73).sin().abs()*2.0;
   let p=vec3(x-width,y+b*0.58,z+b*0.5);
   let q=vec3(x+width,y+b*0.58,z+b*0.5);
   let color=if band%2==0 {grey} else {silver};
   draw_line_3d(p,q,color);
   draw_line_3d(p+vec3(1.2,0.65,0.),q-vec3(1.2,-0.65,0.),color);
  }
 }
}
// Salmon-shaped swimming fish under the frozen lake surface.
fn draw_under_ice_salmon(t:f32,bass:f32,eco:bool) {
 // Continuous travel with stable speed and soft undulating bodies.
 let count=if eco {5} else {12};
 for i in 0..count {
  let f=i as f32;
  let speed=0.75+(f*2.1).sin().abs()*0.28; // modestly faster, no sudden beat jumps
  let z=(f*13.7+t*speed+90.0).rem_euclid(180.0)-90.0;
  let x=(f*3.17).sin()*8.5+(t*0.16+f).sin()*0.20;
  let y=-1.5-(f*1.9).sin().abs()*0.85+(t*0.7+f).sin()*0.06;
  let size=0.6+(f*2.13).sin().abs()*0.55;
  let phase=t*(2.9+speed*0.65)+f*1.3;
  let sway=phase.sin()*size*0.24;
  let nose=vec3(x,y,z+size*1.2);
  let shoulder=vec3(x+phase.sin()*size*0.045,y,z+size*0.4);
  let mid=vec3(x+phase.sin()*size*0.10,y,z-size*0.35);
  let tail=vec3(x+sway,y,z-size*1.35);
  let silver=Color::new(0.56,0.66,0.66,1.);
  let salmon=Color::new(0.69,0.28,0.19,1.);
  let dark=Color::new(0.29,0.37,0.39,1.);
  draw_line_3d(nose,shoulder,silver);
  draw_line_3d(shoulder,mid,salmon);
  draw_line_3d(mid,tail,salmon);
  for side in [-1.0_f32,1.0] {
   let flank=shoulder+vec3(side*size*0.32,0.,-size*0.2);
   draw_line_3d(nose,flank,silver);
   draw_line_3d(flank,mid,salmon);
   draw_line_3d(mid,tail,silver);
  }
  let tail_tip=tail+vec3(sway*0.35,0.,-size*0.65);
  draw_line_3d(tail,tail_tip+vec3(size*0.42,size*0.32,0.),salmon);
  draw_line_3d(tail,tail_tip+vec3(-size*0.42,-size*0.32,0.),salmon);
  draw_line_3d(mid,mid+vec3(0.,size*0.42,-size*0.10),dark);
  // Rare, faint wake rings near the ice surface; no sudden position changes.
  if i%3==0 {
   let cycle=(t*0.34+f*0.31).rem_euclid(3.6);
   if cycle<1.25 {
    let radius=0.10+cycle*0.25;
    let wake=vec3(tail.x,-0.73,tail.z-0.4-cycle*0.6);
    let fade=1.0-cycle/1.25;
    let c=Color::new(0.15*fade,0.35*fade,0.40*fade,1.);
    let steps=if eco {8} else {12};
    for k in 0..steps {
     let a=std::f32::consts::TAU*k as f32/steps as f32;
     let b=std::f32::consts::TAU*(k+1) as f32/steps as f32;
     draw_line_3d(wake+vec3(a.cos()*radius,0.,a.sin()*radius),
                  wake+vec3(b.cos()*radius,0.,b.sin()*radius),c);
    }
   }
  }
  let _=bass;
 }
}

// Stable seeded placement: no per-frame random jitter.
fn v35_fire_rocks_and_reeds(t:f32,bass:f32,eco:bool) {
 let count=if eco {11} else {22};
 for i in 0..count {
  let f=i as f32;
  let a=f*2.399963;
  let distance=13.0+(f*7.39).rem_euclid(42.0);
  let x=a.cos()*distance;
  let z=a.sin()*distance;
  let y=fire_hillside_height(x,z);
  let r=2.2+(f*1.77).sin().abs()*4.2;
  let height=r*(0.65+(f*1.19).cos().abs()*0.6);
  let rock=Color::new(0.25+(f*0.13).sin().abs()*0.12,0.23,0.20,1.);
  // A low-poly angular boulder, broad at the base, narrower on top.
  for j in 0..8 {
   let aa=std::f32::consts::TAU*j as f32/8.;
   let bb=std::f32::consts::TAU*(j+1) as f32/8.;
   let pa=vec3(x+aa.cos()*r,y,z+aa.sin()*r);
   let pb=vec3(x+bb.cos()*r,y,z+bb.sin()*r);
   let apex=vec3(x+r*0.14,y+height,z-r*0.12);
   draw_line_3d(pa,pb,rock);
   draw_line_3d(pa,apex,rock);
   if !eco {draw_line_3d(pa,vec3(x+aa.cos()*r*0.75,y+height*0.45,z+aa.sin()*r*0.75),rock);}
  }
 }
 // Clumps of reeds, swaying smoothly to the music near the rocks and camp.
 for i in 0..(if eco {40} else {95}) {
  let f=i as f32;
  let x=(f*17.11).sin()*45.;
  let z=(f*9.37).cos()*42.;
  if x*x+z*z<45. {continue;}
  let ground=fire_hillside_height(x,z);
  let h=0.65+(f*1.71).sin().abs()*1.8;
  let sway=(t*1.35+f*0.8).sin()*(0.25+bass*0.55);
  let col=if i%3==0 {Color::new(0.56,0.46,0.19,1.)} else {Color::new(0.31,0.36,0.14,1.)};
  draw_line_3d(vec3(x,ground,z),vec3(x+sway,ground+h,z+0.15*sway),col);
  if !eco {draw_line_3d(vec3(x+0.16,ground,z),vec3(x+0.16-sway*0.5,ground+h*0.7,z),col);}
 }
}
fn v35_snow_grey_grass(t:f32,bass:f32,mid:f32,eco:bool) {
 // Grey alpine grass growing on mountain slopes, not on the frozen lake.
 for i in 0..(if eco {75} else {185}) {
  let f=i as f32;
  let side=if i%2==0 {-1.} else {1.};
  let x=side*(17.0+(f*4.61).rem_euclid(37.));
  let z=-36.0+(f*8.29).rem_euclid(73.);
  let y=-0.25+(x.abs()-17.0)*0.105+(z*0.13).sin()*0.48;
  let height=0.35+(f*1.89).sin().abs()*1.25;
  let wind=(t*1.3+f*0.91).sin()*(0.10+0.32*bass+0.15*mid);
  let grey=0.36+(f*0.41).sin().abs()*0.22;
  let c=Color::new(grey*0.93,grey,grey*1.05,1.);
  let root=vec3(x,y,z);
  draw_line_3d(root,root+vec3(wind,height,z.signum()*0.05),c);
  if !eco {draw_line_3d(root,root+vec3(-0.22+wind*0.7,height*0.75,0.10),c);}
 }
}
fn v35_ocean_shark(t:f32,bass:f32,eco:bool) {
 // Smooth continuous orbit. Shark stays beneath the ocean surface.
 let theta=t*0.21;
 let center=vec3(theta.cos()*15.0,-5.5+(t*0.42).sin()*0.45,theta.sin()*12.0);
 let forward=vec3(-theta.sin(),0.,theta.cos());
 let right=vec3(forward.z,0.,-forward.x);
 let grey=Color::new(0.38,0.55,0.62,1.);
 let belly=Color::new(0.63,0.75,0.78,1.);
 let dark=Color::new(0.18,0.30,0.37,1.);
 let len=4.2;
 let nose=center+forward*len;
 let shoulder=center+forward*2.0;
 let middle=center-forward*0.7;
 let tail_root=center-forward*3.6;
 let tail_sway=(t*3.3).sin()*0.65;
 let tail=tail_root+right*tail_sway;
 // Spindle body with a distinct white underside.
 for side in [-1.0_f32,1.0] {
  let flank=shoulder+right*side*1.15;
  let flank2=middle+right*side*1.0;
  draw_line_3d(nose,flank,grey);
  draw_line_3d(flank,flank2,grey);
  draw_line_3d(flank2,tail,grey);
  draw_line_3d(nose,center+right*side*0.4+vec3(0.,-0.75,0.),belly);
  draw_line_3d(center+right*side*0.4+vec3(0.,-0.75,0.),tail,belly);
  let fin=middle+right*side*3.0+vec3(0.,-0.65,0.);
  draw_line_3d(flank2,fin,grey);
  draw_line_3d(fin,middle-forward*1.0+right*side*0.8,grey);
  // Eyes just behind the snout.
  let eye=shoulder+right*side*0.86+vec3(0.,0.14,0.);
  draw_sphere(eye,0.11,None,dark);
  for g in 0..3 {
   let gp=middle+forward*(g as f32*0.3)+right*side*0.98;
   draw_line_3d(gp+vec3(0.,0.15,0.),gp+vec3(0.,-0.25,0.),dark);
  }
 }
 // Tall dorsal fin and smaller rear fin.
 let fin_top=center+forward*0.15+vec3(0.,2.3,0.);
 draw_line_3d(shoulder,fin_top,grey);
 draw_line_3d(fin_top,middle-forward*0.65,grey);
 draw_line_3d(middle-forward*1.5,middle-forward*1.8+vec3(0.,0.75,0.),grey);
 // Animated crescent tail, moving sideways instead of teleporting.
 let upper=tail-forward*0.75+vec3(0.,2.0,0.);
 let lower=tail-forward*0.55+vec3(0.,-1.55,0.);
 draw_line_3d(tail,upper,grey);
 draw_line_3d(tail,lower,grey);
 draw_line_3d(upper,tail-forward*0.3,grey);
 draw_line_3d(lower,tail-forward*0.3,grey);
 if !eco {
  for k in 0..6 {
   let f=k as f32;
   let p=center-forward*(4.0+f*0.9)+vec3(0.,0.2+(t*0.8+f).sin()*0.12,0.);
   let radius=0.08+0.06*(1.+(t*1.2+f).sin());
   draw_line_3d(p+right*radius,p-right*radius,belly);
  }
 }
 let _=bass;
}
fn fire_hillside_height(x:f32,z:f32)->f32 {
 -0.30 + (x+z*0.18).max(0.0)*0.09
 + 0.70*(x*0.085).sin() + 0.45*(z*0.10).cos()
 + 0.19*(x*0.31+z*0.22).sin()
}
fn draw_fire_forest(t:f32,bass:f32,eco:bool) {
 let step=if eco {4.5_f32} else {2.8_f32};
 let n=if eco {10} else {17};
 // Broad, sloping hillside with uneven grass and soil contours.
 for ix in -n..n {
  let x=ix as f32*step;
  for iz in -n..n {
   let z=iz as f32*step;
   let y=fire_hillside_height(x,z);
   let green=0.19+0.12*((x*0.37+z*0.29).sin()*0.5+0.5);
   let c=Color::new(0.14,green,0.08,1.);
   draw_line_3d(vec3(x,y,z),vec3(x+step,fire_hillside_height(x+step,z),z),c);
   draw_line_3d(vec3(x,y,z),vec3(x,fire_hillside_height(x,z+step),z+step),c);
  }
 }
 // Varied hillside trees: olive, green, gold, amber; flexible canopy tips.
 for i in 0..(if eco {20} else {43}) {
  let f=i as f32;
  let a=f*2.39996;
  let radius=19.0+(f*3.13).sin().abs()*49.0;
  let x=a.cos()*radius;
  let z=a.sin()*radius;
  let height=6.0+(f*2.37).sin().abs()*19.0;
  let ground=fire_hillside_height(x,z);
  let trunk=Color::new(0.31,0.20,0.11,1.);
  let variation=(f*3.19).sin()*0.5+0.5;
  let foliage=match i%6 {
   0=>Color::new(0.16,0.34+variation*0.12,0.08,1.), // forest green
   1=>Color::new(0.31,0.35+variation*0.09,0.09,1.), // olive
   2=>Color::new(0.60,0.41+variation*0.12,0.06,1.), // amber
   3=>Color::new(0.68,0.54+variation*0.12,0.10,1.), // yellow
   4=>Color::new(0.48,0.25+variation*0.10,0.06,1.), // autumn orange
   _=>Color::new(0.07,0.28+variation*0.12,0.12,1.),
  };
  let base=vec3(x,ground,z);
  // Continuous sinusoidal canopy sway, stronger on bass but never jumps.
  let wind=(t*0.85+f*1.9).sin()*(0.35+bass*0.60);
  let gust=(t*1.45+f*0.77).sin()*0.15;
  let bend=vec3(wind+gust,0.,wind*0.26);
  let top=base+vec3(0.,height,0.)+bend;
  draw_line_3d(base,base+vec3(0.,height*0.55,0.),trunk);
  draw_line_3d(base+vec3(0.,height*0.55,0.),top,trunk);
  let levels=if eco {3} else {5};
  for level in 0..levels {
   let l=level as f32;
   let fraction=0.27+l*(0.57/levels as f32);
   let spread=(3.4-l*0.48)*(0.75+variation*0.50)*(height/12.0);
   let sway=bend*fraction;
   let branch=base+vec3(0.,height*fraction,0.)+sway;
   let tip=base+vec3(0.,height*(fraction+0.28),0.)+bend*(fraction+0.22);
   for dir in 0..5 {
    let ang=std::f32::consts::TAU*dir as f32/5.0+f;
    let end=branch+vec3(ang.cos()*spread,-spread*0.24,ang.sin()*spread);
    draw_line_3d(tip,end,foliage);
    if !eco {draw_line_3d(branch,end,foliage);}
   }
  }
 }
 // Props: stacked firewood, log benches, hanging cooking pot and campsite stones.
 for side in [-1.0_f32,1.0] {
  let x=side*8.0;
  draw_cube(vec3(x,0.55,3.),vec3(4.2,0.8,1.5),None,Color::new(0.31,0.20,0.10,1.));
  draw_cube_wires(vec3(x,0.55,3.),vec3(4.2,0.8,1.5),Color::new(0.55,0.33,0.16,1.));
  for k in 0..5 {
   let z=-7.0+k as f32*1.1;
   draw_line_3d(vec3(x-1.6,0.45,z),vec3(x+1.6,0.45,z),Color::new(0.44,0.26,0.12,1.));
  }
 }
 // Cooking tripod and suspended pot.
 let iron=Color::new(0.38,0.40,0.40,1.);
 let top=vec3(0.,6.5,0.);
 for k in 0..3 {
  let a=std::f32::consts::TAU*k as f32/3.0;
  draw_line_3d(top,vec3(a.cos()*4.5,0.,a.sin()*4.5),iron);
 }
 draw_line_3d(top,vec3(0.,3.3,0.),iron);
 draw_cube_wires(vec3(0.,3.0,0.),vec3(1.5,1.1,1.5),iron);
 // Wind-blown leaves, pollen and ash across the hillside.
 for i in 0..(if eco {45} else {115}) {
  let f=i as f32;
  let seed=(f*12.989).sin()*78.233;
  let x=(seed*7.1+t*(2.2+bass*2.0)).rem_euclid(90.0)-45.0;
  let z=(f*7.37).sin()*44.0;
  let ground=fire_hillside_height(x,z);
  let y=ground+0.8+(f*3.11+t*0.45).sin().abs()*5.0;
  let c=if i%4==0 {Color::new(0.57,0.38,0.15,1.)}
    else {Color::new(0.24,0.41,0.17,1.)};
  draw_line_3d(vec3(x,y,z),vec3(x-0.28,y-0.06,z+0.08),c);
 }
 // Warm firelight highlights flicker over surrounding stones.
 for k in 0..(if eco {10} else {20}) {
  let f=k as f32;
  let a=f*2.39996;
  let r=5.0+(f*1.3).sin().abs()*7.0;
  let x=a.cos()*r;
  let z=a.sin()*r;
  let glow=0.5+0.5*(t*5.0+f).sin().abs()+bass*0.25;
  draw_line_3d(vec3(x,0.08,z),vec3(x+0.7,0.08,z+0.5),
    Color::new((0.48*glow).min(1.),(0.21*glow).min(1.),0.03,1.));
 }
}
fn draw_campfire(t:f32,bass:f32,treble:f32,eco:bool) {
 let coal=Color::new(0.21,0.11,0.06,1.);
 let ember=Color::new(0.95,0.30,0.045,1.);
 let hot=Color::new(1.0,0.75,0.16,1.);
 // Ring of stones, and crossed chunky faceted wood logs.
 for i in 0..(if eco {9} else {14}) {
  let a=std::f32::consts::TAU*i as f32/(if eco {9.0} else {14.0});
  let p=vec3(a.cos()*3.5,0.15,a.sin()*3.5);
  draw_cube(p,vec3(1.15,0.75,0.9),None,Color::new(0.30,0.28,0.25,1.));
  draw_cube_wires(p,vec3(1.15,0.75,0.9),Color::new(0.48,0.39,0.30,1.));
 }
 for i in 0..7 {
  let a=i as f32*std::f32::consts::TAU/7.0;
  let base=vec3(a.cos()*1.8,0.7,a.sin()*1.8);
  let tip=vec3(-a.cos()*1.1,1.7,-a.sin()*1.1);
  draw_line_3d(base,tip,coal);
  draw_line_3d(base+vec3(0.12,0.,0.12),tip+vec3(0.12,0.,0.12),ember);
  draw_cube_wires(base.lerp(tip,0.5),vec3(0.62,0.68,0.62),coal);
 }
 // Multiple flickering flame tongues, rendered as lightweight tapered strokes.
 let tongues=if eco {13} else {25};
 for i in 0..tongues {
  let f=i as f32;
  let a=f*2.399;
  let r=0.25+(f*4.23).sin().abs()*1.35;
  let flicker=(t*(5.0+treble*5.0)+f*1.73).sin();
  let height=2.3+(f*1.33).sin().abs()*3.0+bass*5.0+flicker*0.8;
  let root=vec3(a.cos()*r,0.8,a.sin()*r);
  let mid=root+vec3(flicker*0.3,height*0.5,(t*3.0+f).cos()*0.35);
  let tip=root+vec3(flicker*0.75,height,(t*2.2+f).sin()*0.65);
  draw_line_3d(root,mid,ember);
  draw_line_3d(mid,tip,hot);
  if !eco {draw_line_3d(root+vec3(0.15,0.,0.),tip,hot);}
 }
 // Smoke rises and disperses with wind, while sparks accelerate on bass.
 let smoke_count=if eco {24} else {58};
 for i in 0..smoke_count {
  let f=i as f32;
  let life=(t*(0.5+(f*1.7).sin().abs()*0.25)+f*0.731).fract();
  let y=3.0+life*19.0;
  let spread=0.6+life*5.0;
  let x=(f*4.19+t*0.7).sin()*spread+life*3.0;
  let z=(f*6.73+t*0.4).cos()*spread;
  let fade=(1.0-life)*0.22;
  let smoke=Color::new(fade,fade*0.97,fade*0.91,1.);
  draw_line_3d(vec3(x,y,z),vec3(x+0.22+life*0.5,y+0.5,z),smoke);
 }
 for i in 0..(if eco {12} else {30}) {
  let f=i as f32;
  let life=(t*(1.5+bass*2.0)+f*0.61).fract();
  let x=(f*2.17).sin()*(0.6+life*3.0);
  let z=(f*3.81).cos()*(0.6+life*3.0);
  let y=1.5+life*12.0;
  draw_line_3d(vec3(x,y,z),vec3(x,y+0.24,z),ember);
 }
 // Expanding circular heat/energy flares, timed to the beat.
 let rings=if eco {3} else {6};
 for i in 0..rings {
  let phase=(t*(0.38+bass*0.65)+i as f32/rings as f32).fract();
  let radius=2.8+phase*14.0;
  let glow=(1.0-phase)*(0.45+bass*0.45);
  let color=Color::new((glow*1.5).min(1.0),glow*0.54,0.045,1.);
  let segments=if eco {20} else {36};
  for j in 0..segments {
   let a=std::f32::consts::TAU*j as f32/segments as f32;
   let b=std::f32::consts::TAU*(j+1) as f32/segments as f32;
   let y=0.07+0.10*(a*3.0+t*2.0).sin();
   draw_line_3d(vec3(a.cos()*radius,y,a.sin()*radius),
                vec3(b.cos()*radius,y,b.sin()*radius),color);
  }
 }

}
// Side-mounted timber jetty with piles, planks, handrails and storm lamps.
// Coastal seawall, landward steps and storm lamp lens flares.
// Breakers hit the seaward face of the concrete wall and spray upward.
// Deterministic phases avoid storing or allocating particles each frame.
fn draw_seawall_spray(t:f32,bass:f32,mid:f32,eco:bool) {
 let sections=if eco {6} else {11};
 for section in 0..sections {
  let f=section as f32;
  let x=3.0+f*(42.0/(sections-1) as f32);
  let z=-33.7;
  let surface=ocean_height(x,z,t,bass,mid);
  // The impact envelope rises and falls smoothly with the swell.
  let crest=(0.5+0.5*(t*1.15+f*0.82).sin()).powf(5.0);
  let force=(0.24+0.76*bass)*crest;
  if force<0.07 {continue;}
  let jets=if eco {5} else {12};
  for j in 0..jets {
   let seed=f*11.37+j as f32*3.71;
   let age=(t*(0.72+(seed*0.2).sin().abs()*0.22)+seed).rem_euclid(1.65);
   if age>1.0 {continue;}
   let lift=force*(3.5+((seed*2.1).sin().abs())*5.0);
   let arc=4.0*age*(1.0-age);
   let dx=(seed*1.8).sin()*1.3*age;
   let dz=(seed*2.4).cos()*1.1*age+age*1.5;
   let y=surface+0.3+arc*lift;
   let p=vec3(x+dx,y,z+dz);
   let color=if j%3==0 {Color::new(0.72,0.91,0.98,1.)}
    else {Color::new(0.33,0.66,0.82,1.)};
   draw_line_3d(p,p+vec3(0.07,0.18+force*0.20,0.11),color);
  }
  // Foamy whitewash at the wall's waterline.
  let foam=Color::new(0.53,0.80,0.89,1.);
  for k in 0..(if eco {2} else {5}) {
   let u=k as f32;
   let yy=surface+0.1+(t*2.3+u+f).sin().abs()*force*0.45;
   draw_line_3d(vec3(x-1.1,yy,z+0.15+u*0.15),
                vec3(x+1.1,yy,z+0.15+u*0.15),foam);
  }
 }
}
fn draw_jetty_shore(t:f32,bass:f32,eco:bool) {
 let cement=Color::new(0.32,0.35,0.37,1.);
 let seams=Color::new(0.14,0.18,0.20,1.);
 let rail=Color::new(0.47,0.52,0.54,1.);
 // Concrete retaining wall behind the jetty, with horizontal seams.
 let wall_z=-36.0;
 draw_cube(vec3(24.,5.5,wall_z),vec3(46.,12.,3.),None,cement);
 draw_cube_wires(vec3(24.,5.5,wall_z),vec3(46.,12.,3.),seams);
 for i in 0..9 {
  let x=2.0+i as f32*5.5;
  draw_line_3d(vec3(x,-0.3,wall_z+1.6),vec3(x,11.4,wall_z+1.6),seams);
 }
 for level in 1..4 {
  let y=level as f32*2.8;
  draw_line_3d(vec3(1.,y,wall_z+1.6),vec3(47.,y,wall_z+1.6),seams);
 }
 // Stair flight descends from the seawall top to the jetty deck at y=4.4.
 let steps=if eco {9} else {12};
 let start_z=-34.0;
 let finish_z=-28.0;
 for i in 0..steps {
  let u=i as f32/(steps-1) as f32;
  let z=start_z+(finish_z-start_z)*u;
  let y=10.6+(4.4-10.6)*u;
  draw_cube(vec3(24.,y-0.22,z),vec3(5.2,0.42,0.62),None,cement);
  draw_line_3d(vec3(21.4,y,z),vec3(26.6,y,z),rail);
  if i%2==0 {
   for side in [-1.0_f32,1.0] {
    let x=24.0+side*2.7;
    draw_line_3d(vec3(x,y,z),vec3(x,y+2.0,z),rail);
   }
  }
 }
 for side in [-1.0_f32,1.0] {
  draw_line_3d(vec3(24.+side*2.7,12.6,start_z),
               vec3(24.+side*2.7,6.4,finish_z),rail);
 }
 // Warm flare crosses at jetty lamp positions, intensity gently flickers.
 for i in 0..5 {
  let z=-24.0+i as f32*11.0;
  let x=27.1;
  let y=9.7;
  let flicker=(0.85+0.15*(t*3.4+i as f32).sin()+bass*0.12).min(1.0);
  let bright=Color::new(flicker,0.66*flicker,0.27*flicker,1.);
  let reach=if eco {0.8} else {1.5};
  draw_line_3d(vec3(x-reach,y,z),vec3(x+reach,y,z),bright);
  draw_line_3d(vec3(x,y-reach,z),vec3(x,y+reach,z),bright);
  if !eco {
   draw_line_3d(vec3(x-reach*0.6,y-reach*0.6,z),
                vec3(x+reach*0.6,y+reach*0.6,z),bright);
   draw_line_3d(vec3(x-reach*0.6,y+reach*0.6,z),
                vec3(x+reach*0.6,y-reach*0.6,z),bright);
  }
 }
}
fn draw_ocean_jetty(t:f32,bass:f32,eco:bool) {
 let timber=Color::new(0.29,0.19,0.12,1.);
 let timber_edge=Color::new(0.51,0.34,0.21,1.);
 let wet=Color::new(0.12,0.18,0.21,1.);
 let lamp=Color::new(1.0,0.66,0.23,1.);
 let x=24.0;
 let y=4.4;
 let posts=if eco {7} else {12};
 // Jetty projects along Z, set to one side of the ocean's central viewpoint.
 for i in 0..posts {
  let z=-28.0+i as f32*5.0;
  for side in [-1.0_f32,1.0] {
   let px=x+side*3.1;
   draw_cube(vec3(px,0.0,z),vec3(0.75,10.0,0.75),None,timber);
   draw_cube_wires(vec3(px,0.0,z),vec3(0.75,10.0,0.75),timber_edge);
   draw_line_3d(vec3(px,y,z),vec3(px,y+2.7,z),timber_edge);
   if i>0 {
    draw_line_3d(vec3(px,y+2.7,z-5.0),vec3(px,y+2.7,z),timber_edge);
    draw_line_3d(vec3(px,y+1.5,z-5.0),vec3(px,y+1.5,z),timber_edge);
   }
  }
  // Cross-braced supports beneath the deck.
  if i>0 {
   draw_line_3d(vec3(x-3.1,-2.5,z-5.0),vec3(x+3.1,3.5,z),wet);
   draw_line_3d(vec3(x+3.1,-2.5,z-5.0),vec3(x-3.1,3.5,z),wet);
  }
 }
 // Individual deck planks and subtle reflected lamp shimmer.
 for i in 0..(if eco {27} else {54}) {
  let z=-29.0+i as f32*(56.0/(if eco {27.0} else {54.0}));
  draw_cube(vec3(x,y,z),vec3(6.5,0.28,0.84),None,timber);
  if i%5==0 {draw_line_3d(vec3(x-3.2,y+0.16,z),vec3(x+3.2,y+0.16,z),timber_edge);}
 }
 for i in 0..5 {
  let z=-24.0+i as f32*11.0;
  draw_line_3d(vec3(x+3.1,y,z),vec3(x+3.1,y+5.1,z),timber_edge);
  let shimmer=0.76+0.24*(t*4.0+i as f32).sin().abs()+bass*0.15;
  draw_sphere(vec3(x+3.1,y+5.3,z),0.34,None,
              Color::new((lamp.r*shimmer).min(1.),lamp.g*shimmer,lamp.b,1.));
  // Reflection on moving ocean waves below each light.
  for k in 0..(if eco {3} else {6}) {
   let zz=z+k as f32*0.7;
   let yy=ocean_height(x+3.1,zz,t,bass,0.15)+0.06;
   draw_line_3d(vec3(x+2.5,yy,zz),vec3(x+3.5,yy,zz+0.3),
                Color::new(0.38,0.24,0.11,1.));
  }
 }
}
fn draw_ocean(t:f32,bass:f32,mid:f32,eco:bool,lightning:f32){
    let step=if eco {4.0_f32} else {2.5_f32};
    let n=if eco {12} else {20};
    let blue=Color::new(0.04+lightning*0.55,0.35+lightning*0.5,0.85,1.);
    let crest=Color::new(0.30+lightning*0.6,0.85,1.,1.);
    for iz in -n..=n {
        let z=iz as f32*step;
        for ix in -n..n {
            let x=ix as f32*step;
            let h=ocean_height(x,z,t,bass,mid);
            let h2=ocean_height(x+step,z,t,bass,mid);
            let c=if h>1.3 {crest} else {blue};
            draw_line_3d(vec3(x,h,z),vec3(x+step,h2,z),c);
        }
    }
    for ix in -n..=n {
        let x=ix as f32*step;
        for iz in -n..n {
            let z=iz as f32*step;
            let h=ocean_height(x,z,t,bass,mid);
            let h2=ocean_height(x,z+step,t,bass,mid);
            let c=if h>1.3 {crest} else {blue};
            draw_line_3d(vec3(x,h,z),vec3(x,h2,z+step),c);
        }
    }
    if lightning>0.035 {
        // Jagged lightning forks with deterministic positions: no allocations.
        let seed=(t*0.6).floor();
        let x=(seed*2.17).sin()*17.0;
        let z=-12.0+(seed*1.31).cos()*8.0;
        let intensity=(lightning*1.4).min(1.0);
        let bright=Color::new(0.65*intensity,0.87*intensity,intensity,1.0);
        let halo=Color::new(0.16*intensity,0.35*intensity,0.75*intensity,1.0);
        for k in 0..9 {
            let f=k as f32;
            let f2=(k+1) as f32;
            let p=vec3(x+(f*2.73+seed).sin()*2.0,23.0-f*2.7,z+(f*1.73).cos()*1.3);
            let q=vec3(x+(f2*2.73+seed).sin()*2.0,23.0-f2*2.7,z+(f2*1.73).cos()*1.3);
            draw_line_3d(p,q,bright);
            if !eco {
                draw_line_3d(p+vec3(0.13,0.,0.),q+vec3(0.13,0.,0.),halo);
            }
            if k==3 || k==5 {
                let branch_end=q+vec3((f*3.1).sin()*6.0,-5.0,(f*1.9).cos()*4.0);
                draw_line_3d(q,branch_end,bright);
                draw_line_3d(branch_end,branch_end+vec3(2.0,-3.0,1.0),halo);
            }
        }
    }
}
// Three-dimensional binary rain inside a lightweight industrial data foundry.
// Each digit is drawn with 3D line segments, not a screen-space text overlay.
fn binary_digit(pos:Vec3, one:bool, size:f32, color:Color) {
 // Rounded stroke-like glyphs: smooth curves made of short connected 3D lines.
 // Consistent width and stable bit identity prevent distracting visual flicker.
 let segments=if one {5} else {12};
 if one {
  let pts=[vec3(-0.19,0.74,0.),vec3(0.05,1.0,0.),vec3(0.05,0.0,0.),vec3(-0.23,0.0,0.),vec3(0.27,0.0,0.)];
  for i in 0..pts.len()-1 {draw_line_3d(pos+pts[i]*size,pos+pts[i+1]*size,color);}
 } else {
  let rx=0.34*size;let ry=0.50*size;
  for i in 0..segments {
   let a=std::f32::consts::TAU*i as f32/segments as f32;
   let b=std::f32::consts::TAU*(i+1) as f32/segments as f32;
   draw_line_3d(pos+vec3(rx*a.cos(),ry+ry*a.sin(),0.),
                pos+vec3(rx*b.cos(),ry+ry*b.sin(),0.),color);
  }
 }
}
// Architectural modules: raised walkways, braced support frames and
// modular equipment racks, all drawn with inexpensive fixed geometry.
fn draw_foundry_architecture(t:f32,bass:f32,eco:bool) {
 let steel=Color::new(0.08,0.14,0.13,1.);
 let outline=Color::new(0.16,0.32,0.25,1.);
 let led=Color::new(0.05,0.55+bass*0.4,0.23,1.);
 // Walkways on both sides with supporting legs and guard rails.
 for side in [-1.0_f32,1.0] {
  let x=side*21.0;
  // Walkway is a transparent wire-mesh platform.
  draw_cube_wires(vec3(x,6.0,0.),vec3(6.,0.6,58.),outline);
  for z_i in -3..=3 {
   let z=z_i as f32*9.0;
   draw_line_3d(vec3(x-side*2.65,6.4,z),vec3(x-side*2.65,9.2,z),outline);
   draw_line_3d(vec3(x+side*2.65,6.4,z),vec3(x+side*2.65,9.2,z),outline);
   draw_line_3d(vec3(x,5.7,z),vec3(x,0.,z),outline);
   if !eco {
    draw_line_3d(vec3(x,5.7,z),vec3(x+side*3.,0.,z+4.),outline);
   }
  }
  for rail_y in [7.7_f32,9.2] {
   draw_line_3d(vec3(x-side*2.65,rail_y,-29.),vec3(x-side*2.65,rail_y,29.),outline);
   draw_line_3d(vec3(x+side*2.65,rail_y,-29.),vec3(x+side*2.65,rail_y,29.),outline);
  }
  // Access stairs are aligned with the OUTER edge of each raised catwalk.
  for end in [-1.0_f32,1.0] {
   for k in 0..5 {
    let f=k as f32;
    let z=end*(29.0+f*1.6);
    let y=5.8-f*1.2;
    draw_cube(vec3(x,y,z),vec3(5.6,0.35,1.45),None,steel);
    for edge in [-2.55_f32,2.55] {
     let px=x+edge;
     draw_line_3d(vec3(px,y+0.2,z),vec3(px,y+2.1,z),outline);
     if k<4 {
      draw_line_3d(vec3(px,y+2.1,z),vec3(px,y+0.9,z+end*1.6),outline);
     }
    }
   }
  }
  signal_line(vec3(x,6.37,-29.),vec3(x,6.37,29.),t,0.13+bass*0.2,side*0.3,bass);
 }
 // Central processor towers and animated cooling turbines.
 for (i,z) in [-18.0_f32,0.0,18.0].iter().enumerate() {
  let x=if i==1 {0.0} else {7.0};
  // Open processor cages instead of solid blocks.
  draw_cube_wires(vec3(x,6.0,*z),vec3(4.,11.,4.),outline);
  for j in 0..4 {
   let y=2.5+j as f32*2.2;
   draw_line_3d(vec3(x-2.1,y,*z+2.1),vec3(x+2.1,y,*z+2.1),led);
  }
  let center=vec3(x,12.6,*z);
  let blades=if eco {4} else {6};
  for blade in 0..blades {
   let angle=t*(0.9+bass*3.0)+std::f32::consts::TAU*blade as f32/blades as f32;
   let tip=center+vec3(angle.cos()*2.3,0.,angle.sin()*2.3);
   draw_line_3d(center,tip,led);
  }
 }
 // Cylindrical ventilating ducts suspended above the server aisles.
 for side in [-1.0_f32,1.0] {
  for z in [-18.0_f32,0.0,18.0] {
   draw_duct_fan(vec3(side*13.0,15.8,z),2.2,6.5,t,bass,eco);
  }
 }
 // Overhead truss lattice and ceiling conduit runs.
 for z_i in -3..=3 {
  let z=z_i as f32*9.0;
  draw_line_3d(vec3(-27.,21.,z),vec3(27.,21.,z),outline);
  draw_line_3d(vec3(-27.,18.,z),vec3(27.,18.,z),outline);
  if !eco {
   for i in -4..4 {
    let x=i as f32*6.0;
    draw_line_3d(vec3(x,18.,z),vec3(x+6.,21.,z),outline);
   }
  }
 }
 for x in [-24.0_f32,-12.0,12.0,24.0] {
  signal_line(vec3(x,20.,-29.),vec3(x,20.,29.),t,0.09+bass*0.18,x*0.07,bass);
 }
}
// Traveling signal packet along a fixed industrial conductor. Draws only a
// short bright segment and a dim trail, avoiding expensive full-screen bloom.
fn signal_line(a:Vec3,b:Vec3,t:f32,speed:f32,phase:f32,bass:f32) {
 let delta=b-a;
 let base=Color::new(0.02,0.16,0.075,1.0);
 draw_line_3d(a,b,base);
 let travel=(t*speed+phase).rem_euclid(1.0);
 let tail=(0.13+bass*0.14).min(0.35);
 let from=(travel-tail).max(0.0);
 let bright=(0.65+bass*0.35).min(1.0);
 draw_line_3d(a+delta*from,a+delta*travel,
   Color::new(0.12,bright,0.27,1.0));
 if travel>0.02 {
  let head=(travel-0.025).max(0.0);
  draw_line_3d(a+delta*head,a+delta*travel,
    Color::new(0.62,1.0,0.78,1.0));
 }
}
// A square wave radiates across the foundry floor on bass hits.
fn signal_floor_pulse(t:f32,bass:f32,eco:bool) {
 let cycle=(t*(0.33+bass*0.35)).fract();
 let radius=2.0+cycle*26.0;
 let intensity=(1.0-cycle)*0.65+bass*0.3;
 let c=Color::new(0.04,intensity.min(1.0),0.14,1.0);
 let count=if eco {12} else {24};
 for i in 0..count {
  let a=std::f32::consts::TAU*i as f32/count as f32;
  let b=std::f32::consts::TAU*(i+1) as f32/count as f32;
  draw_line_3d(vec3(a.cos()*radius,-0.82,a.sin()*radius),
               vec3(b.cos()*radius,-0.82,b.sin()*radius),c);
 }
}
fn draw_matrix_foundry(t:f32,bass:f32,treble:f32,eco:bool) {
 let metal=Color::new(0.10,0.17,0.15,1.);
 let edges=Color::new(0.13,0.36,0.23,1.);
 let neon=Color::new(0.08,0.85,0.30,1.);
 // Industrial floor, gantries, beams, cable trays and machine towers.
 let span=if eco {3} else {2};
 for i in (-10..=10).step_by(span) {
  let v=i as f32*3.0;
  draw_line_3d(vec3(v,-1.,-30.),vec3(v,-1.,30.),edges);
  draw_line_3d(vec3(-30.,-1.,v),vec3(30.,-1.,v),edges);
 }
 for side in [-1.0_f32,1.0] {
  for depth in [-24.0_f32,-8.0,8.0,24.0] {
   let x=side*26.0;
   // Transparent structural steel: retain visible edges without opaque cladding.
  let _=metal;
   draw_cube_wires(vec3(x,6.0,depth),vec3(2.0,12.0,3.5),edges);
   draw_line_3d(vec3(x,12.,depth),vec3(x,19.,depth),edges);
   draw_line_3d(vec3(x,19.,depth),vec3(-x,19.,depth),edges);
   draw_line_3d(vec3(x,14.,depth),vec3(x,14.,depth+7.),neon);
  }
  for row in 0..4 {
   let y=3.0+row as f32*3.2;
   draw_line_3d(vec3(side*23.,y,-30.),vec3(side*23.,y,30.),edges);
  }
 }
 for z in [-20.0_f32,0.0,20.0] {
  // Wireframe data terminal plinth.
  draw_cube_wires(vec3(0.,0.0,z),vec3(5.,2.,5.),neon);
  draw_line_3d(vec3(-4.,3.,z),vec3(4.,3.,z),neon);
 }
 // Neon signal pulses travel through the floor conduits, overhead
 // gantries, wall rails and machine edges, following actual geometry.
 let lanes=if eco {3} else {6};
 for i in 0..lanes {
  let offset=(i as f32-(lanes-1) as f32*0.5)*7.5;
  signal_line(vec3(offset,-0.78,-29.),vec3(offset,-0.78,29.),
              t,0.12+bass*0.25,i as f32*0.23,bass);
  signal_line(vec3(-29.,-0.77,offset),vec3(29.,-0.77,offset),
              t,0.15+bass*0.20,i as f32*0.19,bass);
 }
 for side in [-1.0_f32,1.0] {
  for (i,z) in [-24.0_f32,-8.0,8.0,24.0].iter().enumerate() {
   let x=side*26.0;
   let phase=i as f32*0.21+if side>0.0 {0.5} else {0.0};
   signal_line(vec3(x,0.0,*z),vec3(x,19.0,*z),
               t,0.28+bass*0.28,phase,bass);
   signal_line(vec3(x,19.05,*z),vec3(-x,19.05,*z),
               t,0.18+bass*0.35,phase+0.3,bass);
  }
  for row in 0..(if eco {2} else {4}) {
   let y=3.0+row as f32*3.2;
   signal_line(vec3(side*23.,y,-30.),vec3(side*23.,y,30.),
               t,0.11+bass*0.2,row as f32*0.25,bass);
  }
 }
 for (i,z) in [-20.0_f32,0.0,20.0].iter().enumerate() {
  signal_line(vec3(-2.5,1.08,*z-2.5),vec3(2.5,1.08,*z-2.5),
              t,0.65+bass*0.45,i as f32*0.3,bass);
  signal_line(vec3(2.5,1.08,*z-2.5),vec3(2.5,1.08,*z+2.5),
              t,0.55+bass*0.5,i as f32*0.27,bass);
 }
 signal_floor_pulse(t,bass,eco);
 draw_foundry_architecture(t,bass,eco);
 draw_server_room(t,bass,eco);
 // Several planes at different depths produce true 3D parallax when orbiting.
 let cols=if eco {12} else {20};
 let rows=if eco {8} else {12};
 for layer in 0..3 {
  let z=-18.0+layer as f32*18.0;
  for col in 0..cols {
   let x=(col as f32-(cols-1) as f32*0.5)*2.3;
   let seed=col as f32*13.17+layer as f32*9.43;
   let speed=3.2+(seed.sin().abs()*2.5)+bass*6.0;
   let head=(t*speed+seed*3.7).rem_euclid(33.0);
   for row in 0..rows {
    let y=19.0-head+row as f32*2.0;
    if !(-0.5..=20.0).contains(&y) {continue;}
    // Stable bits change only when their stream advances by one cell.
    let stream_cell=((t*speed+seed*3.7)/2.0).floor() as i32;
    let hash=(col as i32*73+row as i32*37+layer as i32*101+stream_cell*17).wrapping_mul(1103515245);
    let bit=(hash&0x10000)!=0;
    let strength=(1.0-row as f32/rows as f32).max(0.18);
    let color=if row==0 {Color::new(0.65,1.,0.75,1.)}
     else {Color::new(0.02,((0.4+treble*0.4)*strength).min(1.),0.09,1.)};
    binary_digit(vec3(x,y,z),bit,1.25,color);
   }
  }
 }
}
// Server racks, aisle lights and animated status indicators.
// Lightweight circular air duct with visible open ends and rotating fan blades.
fn draw_duct_fan(center:Vec3,r:f32,length:f32,t:f32,bass:f32,eco:bool) {
 let steel=Color::new(0.21,0.40,0.36,1.0);
 let glow=Color::new(0.08,0.84,0.38,1.0);
 let segments=if eco {10} else {18};
 for i in 0..segments {
  let a=std::f32::consts::TAU*i as f32/segments as f32;
  let b=std::f32::consts::TAU*(i+1) as f32/segments as f32;
  let p=vec3(a.cos()*r,a.sin()*r,0.);
  let q=vec3(b.cos()*r,b.sin()*r,0.);
  for z in [-length*0.5,length*0.5] {
   draw_line_3d(center+p+vec3(0.,0.,z),center+q+vec3(0.,0.,z),steel);
  }
  if i%2==0 {draw_line_3d(center+p+vec3(0.,0.,-length*0.5),center+p+vec3(0.,0.,length*0.5),steel);}
 }
 let blades=if eco {4} else {6};
 for i in 0..blades {
  let a=t*(1.2+bass*5.0)+std::f32::consts::TAU*i as f32/blades as f32;
  draw_line_3d(center,center+vec3(a.cos()*r*0.82,a.sin()*r*0.82,0.),glow);
 }
}
// A wire-framed terminal with moving diagnostic traces; no font or texture allocation.
fn draw_lab_monitor(center:Vec3,t:f32,bass:f32,eco:bool) {
 let frame=Color::new(0.32,0.45,0.52,1.0);
 let blue=Color::new(0.10,0.78,1.0,1.0);
 let red=Color::new(0.9,0.16,0.18,1.0);
 draw_cube(center,vec3(0.25,3.6,5.6),None,Color::new(0.035,0.065,0.09,1.0));
 draw_cube_wires(center,vec3(0.30,3.7,5.7),frame);
 let face=center.x-0.18;
 for row in 0..(if eco {3} else {6}) {
  let y=center.y-1.2+row as f32*0.42;
  let length=1.0+((t*1.2+row as f32*2.0).sin().abs())*2.6;
  draw_line_3d(vec3(face,y,center.z-2.35),vec3(face,y,center.z-2.35+length),if row==0 {red} else {blue});
 }
 for i in 0..(if eco {6} else {12}) {
  let z=center.z-2.1+i as f32*0.36;
  let y=center.y-0.6+(i as f32*1.5+t*2.0).sin()*(0.35+bass*0.5);
  if i>0 {
   let zp=center.z-2.1+(i-1) as f32*0.36;
   let yp=center.y-0.6+((i-1) as f32*1.5+t*2.0).sin()*(0.35+bass*0.5);
   draw_line_3d(vec3(face-0.03,yp,zp),vec3(face-0.03,y,z),blue);
  }
 }
}
// Cable routes with visible travelling packets; all positions computed on demand.
fn draw_energy_cable(a:Vec3,control:Vec3,b:Vec3,t:f32,bass:f32,color:Color,eco:bool) {
 let steps=if eco {9} else {17};
 let mut prev=a;
 for i in 1..=steps {
  let u=i as f32/steps as f32;
  let p=a*(1.-u)*(1.-u)+control*(2.*u*(1.-u))+b*u*u;
  draw_line_3d(prev,p,color);
  prev=p;
 }
 let u=(t*(0.23+bass*0.32)+(a.x+a.z)*0.013).rem_euclid(1.0);
 let p=a*(1.-u)*(1.-u)+control*(2.*u*(1.-u))+b*u*u;
 draw_sphere(p,0.13+bass*0.16,None,Color::new(0.35,1.,0.5,1.));
}
fn draw_matrix_cables(t:f32,bass:f32,eco:bool) {
 let rows=if eco {4} else {7};
 for side in [-1.0_f32,1.0] {
  for j in 0..rows {
   let z=-25.0+j as f32*(50.0/(rows-1) as f32);
   let x=side*16.0;
   for cable in 0..(if eco {2} else {4}) {
    let c=cable as f32;
    let a=vec3(x-side*1.95,2.0+c*1.6,z-1.5+c*0.7);
    let b=vec3(x,17.2,z+0.5+c*0.22);
    let control=vec3(x-side*(3.4+c*0.2),7.0+c*0.8,z+2.8);
    let color=if cable%2==0 {Color::new(0.06,0.40,0.15,1.)}
       else {Color::new(0.12,0.32,0.43,1.)};
    draw_energy_cable(a,control,b,t+c*0.25,bass,color,eco);
   }
  }
 }
}
fn draw_iron_wiring(t:f32,bass:f32,eco:bool) {
 let wire=Color::new(0.46,0.15,0.12,1.);
 for side in [-1.0_f32,1.0] {
  for k in 0..(if eco {4} else {9}) {
   let f=k as f32;
   let z=-20.0+f*5.0;
   let a=vec3(side*21.2,3.0+f%3.0,z);
   let b=vec3(side*20.0,12.5,z+4.0);
   let control=vec3(side*18.5,6.5+(f*1.8).sin()*1.5,z+2.0);
   draw_energy_cable(a,control,b,t+f*0.1,bass,wire,eco);
  }
 }
 // Overhead power bus with travelling cyan pulses.
 for k in 0..(if eco {3} else {6}) {
  let f=k as f32;
  draw_energy_cable(vec3(-19.,13.3,-18.+f*7.0),
      vec3(0.,11.0,-17.+f*7.0),vec3(19.,13.3,-18.+f*7.0),
      t+f*0.15,bass,Color::new(0.24,0.45,0.54,1.),eco);
 }
}
fn draw_server_room(t:f32,bass:f32,eco:bool) {
 let rack=Color::new(0.055,0.075,0.085,1.);
 let trim=Color::new(0.18,0.29,0.29,1.);
 let led=Color::new(0.07,0.7+bass*0.25,0.22,1.);
 let rows=if eco {4} else {7};
 for side in [-1.0_f32,1.0] {
  for j in 0..rows {
   let z=-25.0+j as f32*(50.0/(rows-1) as f32);
   let x=side*16.0;
   // Open wire chassis keeps the data hall transparent.
   let _=rack;
   draw_cube_wires(vec3(x,5.0,z),vec3(3.85,11.05,5.05),trim);
   // Server blades face the aisle; each has a status light.
   let face=x-side*1.96;
   for blade in 0..(if eco {6} else {10}) {
    let y=0.6+blade as f32*(if eco {1.55} else {0.98});
    draw_line_3d(vec3(face,y,z-2.05),vec3(face,y,z+2.05),trim);
    let active=((blade+j*3) as f32*2.4+t*(1.2+bass*2.0)).sin()>-0.2;
    if active {draw_line_3d(vec3(face-side*0.04,y,z-1.7),vec3(face-side*0.04,y,z-1.3),led);}
   }
   // Cable ducts connect each cabinet to the overhead trays.
   draw_line_3d(vec3(x,10.5,z),vec3(x,17.5,z),trim);
  }
  draw_line_3d(vec3(side*16.0,17.5,-28.),vec3(side*16.0,17.5,28.),trim);
 }
 // Overhead data bus and illuminated central aisle.
 for z in (-24..=24).step_by(6) {
  let f=z as f32;
  draw_line_3d(vec3(-17.,17.5,f),vec3(17.,17.5,f),trim);
  draw_line_3d(vec3(-4.,-0.88,f),vec3(4.,-0.88,f),led);
 }
}
// Multi-joint assembly robots working on an elevated central platform.
fn draw_assembly_station(t:f32,bass:f32,mid:f32,eco:bool) {
 let frame=Color::new(0.26,0.32,0.39,1.);
 let red=Color::new(0.78,0.10,0.10,1.);
 let cyan=Color::new(0.24,0.85,1.0,1.);
 let platform=vec3(0.,-1.9,-13.);
 draw_cube(platform,vec3(15.,2.0,12.),None,frame);
 draw_cube_wires(platform,vec3(15.1,2.1,12.1),red);
 for side in [-1.0_f32,1.0] {
  let x=side*6.0;
  let base=vec3(x,0.,-13.);
  let pivot=vec3(x,3.0,-13.);
  let swing=(t*(0.65+mid*1.4)+side).sin();
  let elbow=vec3(x-side*(2.0+swing),5.7+bass*1.2,-12.0+swing);
  let tip=vec3(side*(1.5+swing*0.6),4.2+mid*1.7,-12.0);
  draw_cube(base,vec3(2.1,1.5,2.1),None,red);
  draw_sphere(pivot,0.75,None,frame);
  draw_line_3d(pivot,elbow,red);
  draw_line_3d(elbow,tip,frame);
  draw_sphere(elbow,0.7,None,cyan);
  draw_sphere(tip,0.45,None,cyan);
  for dx in [-0.45_f32,0.45] {
   draw_line_3d(tip,tip+vec3(dx,-0.9,0.45),cyan);
  }
  if !eco {
   draw_line_3d(pivot+vec3(0.15,0.,0.),elbow+vec3(0.15,0.,0.),cyan);
  }
 }
 // Part under assembly: a suspended angular chassis and scanning ring.
 let centre=vec3(0.,4.1,-12.);
 draw_cube_wires(centre,vec3(3.4,5.0,2.0),red);
 draw_cube_wires(centre+vec3(0.,-2.0,0.),vec3(4.7,1.0,2.6),frame);
 let scan_y=1.7+(t*1.7).sin().abs()*5.0;
 for i in 0..4 {
  let a=std::f32::consts::TAU*i as f32/4.0;
  let b=std::f32::consts::TAU*(i+1) as f32/4.0;
  draw_line_3d(vec3(a.cos()*3.0,scan_y,-12.+a.sin()*2.0),
               vec3(b.cos()*3.0,scan_y,-12.+b.sin()*2.0),cyan);
 }
}
// Arc-reactor workshop: line geometry only, designed for older integrated graphics.
fn iron_ring(center:Vec3,r:f32,segments:usize,phase:f32,color:Color) {
 for i in 0..segments {
  let a=std::f32::consts::TAU*i as f32/segments as f32+phase;
  let b=std::f32::consts::TAU*(i+1) as f32/segments as f32+phase;
  draw_line_3d(center+vec3(a.cos()*r,a.sin()*r,0.),center+vec3(b.cos()*r,b.sin()*r,0.),color);
 }
}
fn draw_iron_man_lab(t:f32,bass:f32,mid:f32,treble:f32,eco:bool) {
 let steel=Color::new(0.20,0.25,0.31,1.);
 let red=Color::new(0.80,0.07,0.08,1.);
 let cyan=Color::new(0.18,0.78,1.0,1.);
 let hot=Color::new(0.72,0.94,1.0,1.);
 // Industrial floor, support columns and raised ceiling rails.
 for i in -5..=5 {
  let p=i as f32*5.;
  draw_line_3d(vec3(p,-3.,-25.),vec3(p,-3.,25.),steel);
  draw_line_3d(vec3(-25.,-3.,p),vec3(25.,-3.,p),steel);
 }
 for x in [-19.0_f32,19.] {
  for z in [-19.0_f32,0.,19.] {
   draw_cube(vec3(x,5.,z),vec3(1.5,16.,1.5),None,steel);
   draw_cube_wires(vec3(x,5.,z),vec3(1.6,16.1,1.6),red);
   draw_line_3d(vec3(x,13.,z),vec3(-x,13.,z),steel);
  }
 }
 // Workshop roof framing, wall panels, catwalks and maintenance gantries.
 let wall=Color::new(0.12,0.15,0.19,0.50);
 for side in [-1.0_f32,1.0] {
  let x=side*23.;
  for z in [-22.0_f32,-11.,0.,11.,22.] {
   draw_cube(vec3(x,7.,z),vec3(0.9,14.,1.2),None,wall);
   draw_line_3d(vec3(x,14.,z),vec3(-x,14.,z),steel);
  }
  draw_cube(vec3(side*17.,4.,0.),vec3(4.,0.7,46.),None,wall);
  draw_cube_wires(vec3(side*17.,4.,0.),vec3(4.1,0.8,46.1),red);
  for z in (-20..=20).step_by(5) {
   draw_line_3d(vec3(side*15.,4.6,z as f32),vec3(side*15.,7.,z as f32),steel);
  }
  draw_line_3d(vec3(side*15.,7.,-23.),vec3(side*15.,7.,23.),red);
 }
 for z in (-20..=20).step_by(10) {
  draw_line_3d(vec3(-23.,17.,z as f32),vec3(23.,17.,z as f32),steel);
 }
 // Reinforced workshop walls: cladding, bracing, ducts and service lights.
 let panel=Color::new(0.085,0.11,0.15,0.50);
 let rib=Color::new(0.30,0.34,0.39,1.0);
 for side in [-1.0_f32,1.0] {
  let x=side*24.0;
  for bay in -4..=4 {
   let z=bay as f32*5.3;
   draw_cube(vec3(x,6.0,z),vec3(0.35,11.0,4.9),None,panel);
   draw_cube_wires(vec3(x,6.0,z),vec3(0.38,11.1,4.95),rib);
   draw_line_3d(vec3(x-side*0.3,1.0,z-2.2),vec3(x-side*0.3,11.0,z+2.2),steel);
   draw_line_3d(vec3(x-side*0.3,11.0,z-2.2),vec3(x-side*0.3,1.0,z+2.2),steel);
   // Service panels and animated red warning indicators.
   draw_cube(vec3(x-side*0.45,4.3,z),vec3(0.25,2.2,2.0),None,steel);
   if (t*2.2+bay as f32).sin()>0.0 {
    draw_line_3d(vec3(x-side*0.65,5.1,z-0.65),vec3(x-side*0.65,5.1,z+0.65),red);
   }
  }
  // Parallel overhead ventilation ducts and cable trays.
  for y in [11.8_f32,14.5] {
   draw_cube(vec3(side*21.8,y,0.0),vec3(1.1,0.8,46.0),None,steel);
   draw_cube_wires(vec3(side*21.8,y,0.0),vec3(1.2,0.9,46.1),rib);
  }
  for z in (-20..=20).step_by(8) {
   let z=z as f32;
   draw_line_3d(vec3(side*23.0,15.0,z),vec3(side*17.0,17.0,z),rib);
   draw_line_3d(vec3(side*17.0,17.0,z),vec3(side*11.0,17.0,z),rib);
   draw_line_3d(vec3(side*21.0,12.0,z),vec3(side*21.0,15.0,z),cyan);
  }
 }
 // Back wall, large maintenance door, cross-members and hazard lights.
 for x in (-20..=20).step_by(5) {
  let x=x as f32;
  draw_cube(vec3(x,6.0,-25.0),vec3(4.7,12.0,0.5),None,panel);
  draw_line_3d(vec3(x,0.0,-24.65),vec3(x,12.0,-24.65),rib);
 }
 draw_cube_wires(vec3(0.,5.5,-24.2),vec3(10.0,10.5,0.7),red);
 for y in [2.0_f32,5.0,8.0] {
  draw_line_3d(vec3(-4.5,y,-23.8),vec3(4.5,y,-23.8),steel);
 }
 // Engineering consoles along the interior walls, angled as side-wall displays.
 for side in [-1.0_f32,1.0] {
  for z in [-13.0_f32,0.0,13.0] {
   let x=side*20.8;
   draw_lab_monitor(vec3(x,7.0,z),t+z*0.15,bass,eco);
   draw_line_3d(vec3(x,4.9,z),vec3(x,1.0,z),steel);
  }
 }
 draw_assembly_station(t,bass,mid,eco);
 // Suspended reactor chassis with concentric rotating energy rings.
 let core=vec3(0.,5.,0.);
 let pulse=1.0+bass*0.38+(t*3.).sin()*0.035;
 let seg=if eco {24} else {48};
 for ring in 0..4 {
  let r=(2.2+ring as f32*1.1)*pulse;
  let phase=t*(if ring%2==0 {0.35} else {-0.28});
  iron_ring(core,r,seg,phase,if ring%2==0 {cyan} else {red});
  // Spokes give the impression of mechanical rotating assemblies.
  for k in 0..(if eco {6} else {12}) {
   let a=std::f32::consts::TAU*k as f32/(if eco {6.} else {12.})+phase;
   let p=core+vec3(a.cos()*r,a.sin()*r,0.);
   let q=core+vec3(a.cos()*(r+0.6),a.sin()*(r+0.6),0.);
   draw_line_3d(p,q,hot);
  }
 }
 // Central luminous core (multiple layers rather than an expensive bloom shader).
 draw_sphere(core,0.7+bass*1.2,None,cyan);
 if !eco {draw_sphere_wires(core,1.1+bass*1.5,None,hot);}
 // Reactor coil spokes, responsive to mids.
 for k in 0..(if eco {8} else {16}) {
  let a=std::f32::consts::TAU*k as f32/(if eco {8.} else {16.})+t*0.12;
  let r=1.1+mid*2.0;
  draw_line_3d(core+vec3(a.cos()*r,a.sin()*r,0.),core+vec3(a.cos()*(r+0.8),a.sin()*(r+0.8),0.),hot);
 }
 // Holographic targeting reticle in a separate depth plane.
 let hud=vec3(0.,5.,5.);
 iron_ring(hud,8.0,seg,-t*0.15,Color::new(0.12,0.48,0.7,1.));
 for k in 0..(if eco {8} else {16}) {
  let a=std::f32::consts::TAU*k as f32/(if eco {8.} else {16.})+t*0.08;
  let r=8.0;
  draw_line_3d(hud+vec3(a.cos()*r,a.sin()*r,0.),hud+vec3(a.cos()*(r+0.55+treble),a.sin()*(r+0.55+treble),0.),cyan);
 }
 // Suspended robotic arm segments with music-driven articulation.
 for side in [-1.0_f32,1.0] {
  let shoulder=vec3(side*13.,10.,-1.);
  let elbow=vec3(side*(9.0+mid*2.),7.+(t*0.9).sin(),0.);
  let wrist=vec3(side*(5.8+bass*1.8),5.+(t*1.4).sin()*0.8,1.);
  draw_line_3d(shoulder,elbow,steel);
  draw_line_3d(elbow,wrist,red);
  draw_sphere(elbow,0.55,None,red);
  draw_sphere(wrist,0.4,None,cyan);
 }
 // Pulsing conduits run out from the reactor across the factory floor.
 for k in 0..8 {
  let a=std::f32::consts::TAU*k as f32/8.;
  let end=vec3(a.cos()*19.,-2.8,a.sin()*19.);
  draw_line_3d(vec3(0.,-2.8,0.),end,steel);
  let f=(t*(0.24+bass*0.45)+k as f32*0.13).fract();
  let p=vec3(end.x*f,-2.75,end.z*f);
  let q=vec3(end.x*(f+0.04).min(1.),-2.75,end.z*(f+0.04).min(1.));
  draw_line_3d(p,q,hot);
 }
}

// V26: Rainbow Crystal Dimension. Lightweight wireframe geometry, no textures.
fn rainbow_color(h:f32, brightness:f32)->Color {
 let a=std::f32::consts::TAU*h;
 Color::new(
  ((a).cos()*0.5+0.5)*brightness,
  ((a-std::f32::consts::TAU/3.0).cos()*0.5+0.5)*brightness,
  ((a+std::f32::consts::TAU/3.0).cos()*0.5+0.5)*brightness,1.0)
}
fn rainbow_ring(center:Vec3,radius:f32,angle:f32,segments:usize,phase:f32,brightness:f32) {
 for k in 0..segments {
  let a=std::f32::consts::TAU*k as f32/segments as f32+angle;
  let b=std::f32::consts::TAU*(k+1) as f32/segments as f32+angle;
  let p=center+vec3(a.cos()*radius,a.sin()*radius,0.);
  let q=center+vec3(b.cos()*radius,b.sin()*radius,0.);
  draw_line_3d(p,q,rainbow_color(k as f32/segments as f32+phase,brightness));
 }
}
// V27: a central prismatic reactor with orbiting wireframe rings.
fn draw_prism_reactor(t:f32,bass:f32,mid:f32,treble:f32,eco:bool) {
 let centre=vec3(0.,6.0,0.);
 let pulse=1.0+bass*0.7;
 let rings=if eco {3} else {5};
 let segments=if eco {20} else {36};
 for i in 0..rings {
  let f=i as f32;
  let r=(2.4+f*1.05)*pulse;
  let tilt=t*(0.21+f*0.08);
  let mut last=Vec3::ZERO;
  for k in 0..=segments {
   let a=std::f32::consts::TAU*k as f32/segments as f32;
   let p=centre+vec3(a.cos()*r*tilt.cos(),a.sin()*r,(a.cos()*tilt.sin())*r);
   if k>0 {draw_line_3d(last,p,rainbow_color(f*0.17+a/std::f32::consts::TAU+t*0.045,0.65+treble*0.35));}
   last=p;
  }
 }
 // Suspended double pyramidal energy core.
 let height=2.4+mid*2.0;
 let top=centre+vec3(0.,height,0.);
 let bottom=centre-vec3(0.,height,0.);
 for i in 0..4 {
  let a=std::f32::consts::FRAC_PI_2*i as f32+t*0.22;
  let b=std::f32::consts::FRAC_PI_2*(i+1) as f32+t*0.22;
  let p=centre+vec3(a.cos()*1.6*pulse,0.,a.sin()*1.6*pulse);
  let q=centre+vec3(b.cos()*1.6*pulse,0.,b.sin()*1.6*pulse);
  let color=rainbow_color(i as f32*0.25+t*0.07,1.0);
  draw_line_3d(p,q,color);
  draw_line_3d(p,top,color);
  draw_line_3d(p,bottom,color);
 }
 // Radial reactor spokes pulse outward with bass.
 for i in 0..(if eco {8} else {16}) {
  let a=std::f32::consts::TAU*i as f32/(if eco {8.0} else {16.0})+t*0.09;
  let r=4.0+(t*(2.0+bass*2.0)+i as f32*0.73).rem_euclid(12.0);
  draw_line_3d(centre+vec3(a.cos()*r,0.,a.sin()*r),
    centre+vec3(a.cos()*(r+1.2),0.,a.sin()*(r+1.2)),
    rainbow_color(i as f32*0.13+t*0.05,0.75));
 }
}
// Lightweight prismatic arcs jumping between fixed crystal positions.
fn draw_prismatic_arcs(t:f32,bass:f32,treble:f32,eco:bool) {
 let arcs=if eco {3} else {7};
 for i in 0..arcs {
  let f=i as f32;
  let x=(f*12.71).sin()*22.0;
  let z=(f*29.13).cos()*30.0;
  let next=f+3.0;
  let end_x=(next*12.71).sin()*22.0;
  let end_z=(next*29.13).cos()*30.0;
  let a=vec3(x,3.5+(f*2.37).sin().abs()*3.0,z);
  let b=vec3(end_x,3.5+(next*2.37).sin().abs()*3.0,end_z);
  // Flicker in a slow, deterministic cycle, not every frame.
  let phase=(t*1.6+f*0.71).rem_euclid(3.5);
  if phase>0.9+treble*0.4 && bass<0.25 {continue;}
  let segments=if eco {5} else {9};
  let mut prev=a;
  for j in 1..=segments {
   let u=j as f32/segments as f32;
   let jitter=(u*19.0+f*7.0+(t*8.0).floor()).sin()*0.55;
   let p=a.lerp(b,u)+vec3(jitter,(u*std::f32::consts::PI).sin()*2.5, -jitter*0.65);
   draw_line_3d(prev,p,rainbow_color(f*0.14+t*0.07+u*0.2,0.8+treble*0.2));
   prev=p;
  }
 }
}
// Nested rotating polygons produce a kaleidoscopic portal without shaders.
fn draw_kaleidoscope(t:f32,bass:f32,eco:bool) {
 let centre=vec3(0.,7.0,-33.0);
 for layer in 0..(if eco {3} else {6}) {
  let f=layer as f32;
  let n=if eco {6} else {8};
  let radius=(3.0+f*1.4)*(1.0+bass*0.24);
  let angle=t*(0.14+f*0.035)*if layer%2==0 {1.0} else {-1.0};
  for k in 0..n {
   let a=std::f32::consts::TAU*k as f32/n as f32+angle;
   let b=std::f32::consts::TAU*(k+1) as f32/n as f32+angle;
   let p=centre+vec3(a.cos()*radius,a.sin()*radius,f*0.6);
   let q=centre+vec3(b.cos()*radius,b.sin()*radius,f*0.6);
   draw_line_3d(p,q,rainbow_color(k as f32/n as f32+f*0.12+t*0.03,0.75));
   if !eco {draw_line_3d(p,centre+vec3(0.,0.,f*0.6),rainbow_color(k as f32/n as f32+t*0.02,0.35));}
  }
 }
}
// GPU-friendly simulated lighting: additive-looking layered geometry without shaders.
fn draw_rainbow_lighting(t:f32,bass:f32,mid:f32,treble:f32,eco:bool) {
 let tau=std::f32::consts::TAU;
 // Sweeping searchlight rays emerge from the reactor and wash the valley.
 let origin=vec3(0.,7.0,0.);
 let beams=if eco {5} else {12};
 for i in 0..beams {
  let f=i as f32;
  let angle=t*(0.22+treble*0.14)+tau*f/beams as f32;
  let len=13.0+mid*8.0;
  let end=origin+vec3(angle.cos()*len,-4.2+(t*0.8+f).sin()*2.0,angle.sin()*len);
  let hue=f/beams as f32+t*0.045;
  let c=rainbow_color(hue,0.22+treble*0.30);
  draw_line_3d(origin,end,c);
  if !eco {draw_line_3d(origin+vec3(0.12,0.,0.),end+vec3(0.28,0.,0.),rainbow_color(hue,0.12));}
 }
 // Crystal-tip glints pulse smoothly on high frequencies.
 for i in 0..(if eco {12} else {26}) {
  let f=i as f32;
  let x=(f*12.71).sin()*22.0;
  let z=(f*29.13).cos()*30.0;
  let h=2.5+(f*2.37).sin().abs()*5.0+mid*2.0;
  let tip=vec3(x+(t*0.4+f*1.1).sin()*0.2,h+(t*0.8+f).sin()*0.25,z);
  let twinkle=((t*2.5+f*2.1).sin()*0.5+0.5)*(0.25+treble*0.75);
  let r=0.12+twinkle*0.50;
  let color=rainbow_color(f*0.137+t*0.03,0.42+twinkle*0.45);
  draw_line_3d(tip+vec3(-r,0.,0.),tip+vec3(r,0.,0.),color);
  draw_line_3d(tip+vec3(0.,-r,0.),tip+vec3(0.,r,0.),color);
  if !eco {draw_line_3d(tip+vec3(0.,0.,-r),tip+vec3(0.,0.,r),color);}
 }
 // Orbiting lens-flare crosses around the reactor; bloom impression with layered strokes.
 for i in 0..(if eco {4} else {8}) {
  let f=i as f32;
  let angle=t*0.55+tau*f/(if eco {4.0} else {8.0});
  let r=3.0+bass*2.0;
  let pos=origin+vec3(angle.cos()*r,(t*1.2+f).sin()*1.5,angle.sin()*r);
  let hue=f*0.125+t*0.07;
  let size=0.25+bass*0.65+treble*0.25;
  for layer in 0..(if eco {1} else {3}) {
   let extent=size*(1.0+layer as f32*0.85);
   let brightness=if layer==0 {0.95} else {0.26};
   let c=rainbow_color(hue,brightness);
   draw_line_3d(pos+vec3(-extent,0.,0.),pos+vec3(extent,0.,0.),c);
   draw_line_3d(pos+vec3(0.,-extent,0.),pos+vec3(0.,extent,0.),c);
  }
 }
 // Beat-driven floor light pools below the crystals.
 for i in 0..(if eco {5} else {12}) {
  let f=i as f32;
  let centre=vec3((f*12.71).sin()*22.0,-0.42,(f*29.13).cos()*30.0);
  let radius=1.0+bass*1.8+(t*1.4+f).sin().abs()*0.3;
  let n=if eco {8} else {12};
  for j in 0..n {
   let a=t*0.12+tau*j as f32/n as f32;
   let b=t*0.12+tau*(j+1) as f32/n as f32;
   let c=rainbow_color(f*0.137+t*0.03,0.25+bass*0.4);
   draw_line_3d(centre+vec3(a.cos()*radius,0.,a.sin()*radius),centre+vec3(b.cos()*radius,0.,b.sin()*radius),c);
  }
 }
}
fn draw_rainbow_dimension(t:f32,bass:f32,mid:f32,treble:f32,eco:bool) {
 let seg=if eco {18} else {32};
 // Successive portals recede into the distance, and rotate independently.
 for ring in 0..(if eco {7} else {12}) {
  let f=ring as f32;
  let z=-32.0+f*6.2;
  let r=7.5+bass*2.5+(f*0.6+t*0.6).sin()*0.65;
  let angle=t*(0.11+f*0.012)*if ring%2==0 {1.0} else {-1.0};
  rainbow_ring(vec3(0.,6.0,z),r,angle,seg,t*0.07+f*0.075,0.70+treble*0.30);
  if !eco && ring%3==0 {
   rainbow_ring(vec3(0.,6.0,z),r+0.6,angle+0.06,seg,t*0.07+f*0.075,0.45);
  }
 }
 // V36: six-sided quartz prisms with tapered pyramidal points, bevels,
 // facets, inner refraction lines, and natural crystal clusters.
 // Stable geometry: music changes highlights and subtle height only.
 for i in 0..(if eco {12} else {26}) {
  let f=i as f32;
  let x=(f*12.71).sin()*22.0;
  let z=(f*29.13).cos()*30.0;
  let height=3.0+(f*2.37).sin().abs()*5.5+mid*0.65;
  let width=0.75+(f*1.9).sin().abs()*0.85;
  let color=rainbow_color(f*0.137+t*0.025,0.8);
  let bright=rainbow_color(f*0.137+t*0.025,0.98);
  let shade=rainbow_color(f*0.137+t*0.025,0.34);
  let base=vec3(x,0.,z);
  let angle=f*1.618;
  let prism_height=height*0.72;
  // Deterministic slight tilt: about 4-12 degrees, different for each crystal.
  // Every vertex shares the same lean, so facets remain aligned.
  let lean_angle=(4.0+(f*7.37).sin().abs()*8.0).to_radians();
  let lean_direction=f*2.399963;
  let lean=vec3(lean_direction.cos(),0.,lean_direction.sin())*lean_angle.tan();
  let tilted=|p:Vec3| p+lean*p.y;
  let tip=tilted(base+vec3(0.,height,0.));
  let n=6;
  for side in 0..n {
   let a=std::f32::consts::TAU*side as f32/n as f32+angle;
   let b=std::f32::consts::TAU*(side+1) as f32/n as f32+angle;
   let p=base+vec3(a.cos()*width,0.,a.sin()*width);
   let q=base+vec3(b.cos()*width,0.,b.sin()*width);
   let shoulder=tilted(base+vec3(a.cos()*width,prism_height,a.sin()*width));
   let next=tilted(base+vec3(b.cos()*width,prism_height,b.sin()*width));
   // Real quartz silhouette: hexagonal base, vertical facets, pointed termination.
   draw_line_3d(p,q,shade);
   draw_line_3d(p,shoulder,if side%2==0 {color} else {shade});
   draw_line_3d(shoulder,next,bright);
   draw_line_3d(shoulder,tip,if side%2==0 {bright} else {color});
   if !eco && side%2==0 {
    let glint=shoulder.lerp(tip,0.52+(t*1.2+f).sin()*0.12);
    draw_line_3d(p,glint,shade);
   }
  }
  // Smaller angled satellite crystals form convincing natural clusters.
  if !eco || i%2==0 {
   for satellite in 0..2 {
    let sf=satellite as f32;
    let a=angle+sf*2.5+0.8;
    let foot=base+vec3(a.cos()*width*1.0,0.,a.sin()*width*1.0);
    let h=height*(0.33+sf*0.14);
    let tip2=foot+vec3(a.cos()*h*0.16,h,a.sin()*h*0.16);
    let w=width*0.45;
    for side in 0..4 {
     let theta=std::f32::consts::FRAC_PI_2*side as f32;
     let theta2=theta+std::f32::consts::FRAC_PI_2;
     let p=foot+vec3(theta.cos()*w,0.,theta.sin()*w);
     let q=foot+vec3(theta2.cos()*w,0.,theta2.sin()*w);
     draw_line_3d(p,q,shade);
     draw_line_3d(p,tip2,if side%2==0 {bright} else {color});
    }
   }
  }
 }
 // A liquid-colour floor: intersecting flowing wave strips.
 let spacing=if eco {4.0_f32} else {2.5_f32};
 let n=if eco {9} else {15};
 for k in -n..=n {
  let x=k as f32*spacing;
  let z=k as f32*spacing;
  for j in -n..n {
   let u=j as f32*spacing;
   let wave=|xx:f32,zz:f32| -0.6+(xx*0.17+zz*0.23-t*1.1).sin()*(0.25+bass*1.4);
   let c=rainbow_color((k as f32/n as f32)*0.28+t*0.06+u*0.008,0.35+mid*0.45);
   draw_line_3d(vec3(x,wave(x,u),u),vec3(x,wave(x,u+spacing),u+spacing),c);
   draw_line_3d(vec3(u,wave(u,z),z),vec3(u+spacing,wave(u+spacing,z),z),c);
  }
 }
 // Travelling beat energy along a handful of floor rays.
 for i in 0..(if eco {6} else {12}) {
  let a=std::f32::consts::TAU*i as f32/(if eco {6.0} else {12.0});
  let f=(t*(0.25+bass*0.6)+i as f32*0.21).fract();
  let r=3.0+f*25.0;
  let p=vec3(a.cos()*r,-0.28,a.sin()*r);
  let q=vec3(a.cos()*(r+1.5),-0.28,a.sin()*(r+1.5));
  draw_line_3d(p,q,rainbow_color(i as f32*0.11+t*0.05,1.0));
 }
}

/// How strongly a scene that has not been shown for a while is favoured (per switch waited, up to 10).
/// 0.25 is the smallest value that, in tests, shows all eleven scenes within 40 switches for every
/// kind of steady music tried, while heavy bass still favours Toxic and calm music Snow or Fireflies.
const FAIR_BONUS:f32=0.25;
/// Pull towards each scene's fair share of total screen time over a whole session.
const BALANCE:f32=0.05;
/// Standing preference for the forest scenes (7 Fire, 9 Firefly Forest, 10 Cyber Firefly). Without it,
/// busy or bassy music could go a whole screensaver run without ever showing a forest.
const FOREST_BOOST:f32=0.45;
/// Slow running picture of the music, used by the automatic scene director.
/// These are heuristics over lightweight features, not a genre classifier.
#[derive(Default)]
struct Mood{bass:f32,vocal:f32,bright:f32,density:f32,perc:f32,energy:f32,loud_fast:f32,loud_slow:f32,shown_age:[u32;11],shown_count:[u32;11]}
impl Mood{
 fn update(&mut self,l:&Levels,dt:f32){
  let k=|tau:f32|1.-(-dt/tau).exp();
  let a=k(10.);
  self.bass+=(l.bass-self.bass)*a;
  self.vocal+=(l.vocal-self.vocal)*a;
  self.bright+=(l.brightness-self.bright)*a;
  self.density+=(l.density-self.density)*a;
  self.perc+=(l.kick.max(l.snare)-self.perc)*a;
  self.energy+=(l.energy-self.energy)*a;
  let loud=(l.bass+l.mid+l.treble)/3.;
  self.loud_fast+=(loud-self.loud_fast)*k(1.0);
  self.loud_slow+=(loud-self.loud_slow)*k(8.0);
 }
 /// A breakdown (sudden drop in loudness) or a drop (sudden rise) is a natural place to change scene.
 fn boundary(&self)->bool{
  self.loud_slow>0.05&&(self.loud_fast<self.loud_slow*0.65||self.loud_fast>self.loud_slow*1.45)
 }
 /// A profile that favours no scene, carrying over only the "how long since shown" memory. Used in
 /// silence, where the real profile reads as "perfectly calm" and would keep choosing Snow/Fireflies.
 fn neutral(&self)->Mood{Mood{bass:0.5,vocal:0.5,bright:0.5,density:0.5,perc:0.5,energy:0.5,loud_fast:0.,loud_slow:0.,shown_age:self.shown_age,shown_count:self.shown_count}}
 /// Remember that scene `idx` is on screen now: every other scene has now waited one more switch.
 fn note_shown(&mut self,idx:usize){for a in self.shown_age.iter_mut(){*a+=1;}if idx<11{self.shown_age[idx]=0;self.shown_count[idx]+=1;}}
 /// Best-matching scene for the music, with a growing bonus for scenes that have not been shown for
 /// a while, so that every scene comes round eventually instead of only the best matches.
 fn pick(&self,current:usize,recent:&[usize;3])->usize{self.pick_with(current,recent,FAIR_BONUS)}
 fn pick_with(&self,current:usize,recent:&[usize;3],bonus_w:f32)->usize{self.pick_full(current,recent,bonus_w,BALANCE,FOREST_BOOST)}
 fn pick_full(&self,current:usize,recent:&[usize;3],bonus_w:f32,balance:f32,forest:f32)->usize{
  let calm=1.-self.energy;let (b,v)=(self.bright,self.vocal);
  let score=[
   self.bass*1.2+(1.-b)*0.5+self.energy*0.3,            // 0 Toxic: heavy bass, dark
   self.energy*1.0+self.density*0.6+b*0.4,              // 1 Synthwave: energetic, bright
   self.energy*0.9+self.density*0.9+self.bass*0.2,      // 2 Beat Sync: busy electronic
   v*1.1+b*0.5+(1.-self.perc)*0.4,                      // 3 Rainbow: melodic
   self.perc*1.2+self.density*0.5+(1.-v)*0.3,           // 4 Iron Man: percussive
   calm*1.2+(1.-self.density)*0.6+(1.-self.bass)*0.3,   // 5 Snow: sparse, quiet
   self.bass*0.9+self.density*0.6+(1.-v)*0.3,           // 6 Matrix: driving bass
   (1.-b)*0.8+self.energy*0.4+v*0.2,                    // 7 Fire: warm
   self.bass*0.6+calm*0.4+(1.-self.density)*0.3,        // 8 Ocean: slow and heavy
   calm*0.9+(1.-self.density)*0.4+v*0.3,                // 9 Firefly Forest: atmospheric
   self.energy*0.5+self.density*0.4+(1.-b)*0.2,         // 10 Cyber Firefly
  ];
  let mean_count=self.shown_count.iter().sum::<u32>() as f32/11.;
  let mut best=(current,f32::MIN);
  for (i,sc) in score.iter().enumerate(){
   let mut x=*sc+rand::gen_range(0.,0.25);
   if i==current{x-=2.;}
   if recent.contains(&i){x-=0.8;}
   x+=bonus_w*self.shown_age[i].min(10) as f32;
   // Long-run balance: a scene that has been shown less often than its fair share catches up.
   if bonus_w>0.0 {x+=balance*(mean_count-self.shown_count[i] as f32).clamp(-3.,3.);}
   // The three forest scenes (Fire, Firefly Forest, Cyber Firefly) get a small standing boost.
   if bonus_w>0.0 && (i==7||i==9||i==10) {x+=forest;}
   if x>best.1{best=(i,x);}
  }
  best.0
 }
}
/// Screensaver diagnostics: one line per event in ~/.cache/ben94z/screensaver.log (Linux) or
/// %LOCALAPPDATA%\BEN94Z\screensaver.log (Windows), so "why did it stop?" always has an answer.
fn ss_log(msg:&str){
 use std::io::Write;
 use std::path::PathBuf;
 let dir:Option<PathBuf>=if cfg!(windows){std::env::var("LOCALAPPDATA").ok().map(|d|PathBuf::from(d).join("BEN94Z"))}
  else{std::env::var("XDG_CACHE_HOME").ok().map(|d|PathBuf::from(d).join("ben94z")).or_else(||std::env::var("HOME").ok().map(|h|PathBuf::from(h).join(".cache/ben94z")))};
 if let Some(dir)=dir{
  let _=std::fs::create_dir_all(&dir);
  let p=dir.join("screensaver.log");
  if std::fs::metadata(&p).map(|m|m.len()>200_000).unwrap_or(false){let _=std::fs::remove_file(&p);}
  if let Ok(mut f)=std::fs::OpenOptions::new().create(true).append(true).open(&p){
   let t=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d|d.as_secs()).unwrap_or(0);
   let _=writeln!(f,"[{:02}:{:02}:{:02} UTC] {}",(t/3600)%24,(t/60)%60,t%60,msg);
  }
 }
 eprintln!("BEN94Z screensaver: {}",msg);
}
/// Command-line switches. `--screensaver` (also `-s` and the Windows `/s`) runs fullscreen with no
/// HUD and exits on any key or mouse activity. `/p` (Windows preview) and `/c` (configure) just exit.
fn has_arg(names:&[&str])->bool{std::env::args().skip(1).any(|a|{let a=a.to_lowercase();names.iter().any(|n|a==*n||a.starts_with(&format!("{}:",n)))})}
fn is_screensaver()->bool{has_arg(&["--screensaver","-s","/s"])}
fn window_conf()->Conf{Conf{window_title:"BEN94Z — Rust Audio Visualizer".to_owned(),window_width:960,window_height:540,high_dpi:false,fullscreen:is_screensaver(),..Default::default()}}
#[macroquad::main(window_conf)]
async fn main(){
 if has_arg(&["/p","-p"]) {return;} // Windows screensaver preview pane: nothing to embed
 if has_arg(&["/c","-c"]) {println!("BEN94Z has no settings dialog. Environment variables are listed in README.txt.");return;}
 let screensaver=is_screensaver();
 let (mut ss_age,mut ss_mouse)=(0f32,(0f32,0f32));
 // BEN94Z_MANAGED is set by the idle launcher, which ends the screensaver itself (from GNOME's idle
 // time) when you touch the mouse. Then only keys and clicks end it from inside; mouse *movement*
 // is left to the launcher because window-relative pointer positions are unreliable at startup.
 let managed=std::env::var("BEN94Z_MANAGED").is_ok();
 let mut ss_size=(screen_width(),screen_height());
 let mut ss_size_changed=0f32;
 if screensaver {
  show_mouse(false);
  ss_log(&format!("start: managed={} session={} window={}x{}",managed,std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_|"?".into()),ss_size.0 as i32,ss_size.1 as i32));
 }
 let mut audio=AudioInput::start();
 let mut theme=0usize;
 let mut scene_auto=true; // ENTER toggles automatic scene cycling; enabled at startup.
 let mut scene_timer=0.0_f32;

 let min_dwell:f32=std::env::var("BEN94Z_SCENE_SECS").ok().and_then(|v|v.parse::<f32>().ok()).unwrap_or(30.).clamp(8.,300.);
 let max_dwell=min_dwell*2.5;
 let target_fps:f32=std::env::var("BEN94Z_FPS").ok().and_then(|v|v.parse::<f32>().ok()).unwrap_or(30.).clamp(15.,60.);
 let frame_budget=Duration::from_secs_f32(1./target_fps);
 // DETAIL stays on until you press Q. BEN94Z_AUTO_ECO=1 turns the automatic slow-frame downgrade back on.
 let auto_eco=std::env::var("BEN94Z_AUTO_ECO").is_ok();
 let mut react=0.85f32; // overall reaction strength, [ and ] adjust
 let (mut sb,mut sm,mut st)=(0f32,0f32,0f32);
 let mut spec_s=[0f32;BANDS];
 let mut mood=Mood::default();
 let mut recent=[usize::MAX;3];
 // Seed the random generator (macroquad's default seed is the same every run), then let the
 // screensaver open on a random scene so it does not always start on Toxic.
 rand::srand(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d|d.as_nanos() as u64).unwrap_or(1));
 if screensaver {theme=rand::gen_range(0,THEMES.len() as i32) as usize;ss_log(&format!("starting scene: {}",THEMES[theme]));}
 mood.note_shown(theme);
 let mut pending:Option<usize>=None;
 let mut fade=0f32;
 let mut flash_cool=0f32;
 let mut slow_timer=0f32;
 let mut notice=0f32;
 let mut camera_angle=0f32;
 let mut tilt=22f32;
 let mut distance=52f32;
 let mut auto_rotate=true;
 let mut show_fps=false;
 let mut show_details=!screensaver; // F11 toggles all HUD text; hidden in screensaver mode.
 let mut fullscreen=false; // F12 toggles fullscreen.
 let mut show_grid=true;
 let mut eco_mode=std::env::var("BEN94Z_ECO").is_ok(); // DETAIL by default; BEN94Z_ECO=1 starts in ECO instead.

 let mut synth_scene=synth::Synth::new();
 let mut tunnel=synth::Tunnel::new();
 let (mut since_kick,mut since_snare)=(9f32,9f32);
 let mut last_snare_count=0u32;
  let mut fireflies:Vec<Firefly>=(0..FIREFLY_COUNT).map(|_|Firefly::new()).collect();
 let mut last_kick_count=0u32;
 let mut shockwaves:Vec<(f32,f32)>=Vec::new();
 let mut flash=0f32;
 let mut ocean_flash=0f32;
 let mut ocean_storm=true;
 let mut thunder_timer=0.0_f32;
 let mut time=0f32;
 let mut mouse_last=mouse_position();
 loop {
  let frame_start=Instant::now();
  let dt=get_frame_time().min(0.05);
  time+=dt;
  if screensaver {
   // Real activity ends the screensaver, but only once the window has settled: input in the first
   // 3 s is ignored, and so is anything within 2 s of the window changing size (going fullscreen
   // moves the window-relative pointer position without the user doing anything).
   ss_age+=dt;
   let (w,h)=(screen_width(),screen_height());
   if (w-ss_size.0).abs()>1.0||(h-ss_size.1).abs()>1.0 {ss_size=(w,h);ss_size_changed=ss_age;ss_log(&format!("window size now {}x{}",w as i32,h as i32));}
   let mp=mouse_position();
   if ss_age<3.0 || ss_age-ss_size_changed<2.0 {ss_mouse=mp;let _=get_last_key_pressed();}
   else {
    let moved=(mp.0-ss_mouse.0).abs()+(mp.1-ss_mouse.1).abs();
    let why:Option<String>=if let Some(k)=get_last_key_pressed(){Some(format!("key {:?}",k))}
     else if is_mouse_button_pressed(MouseButton::Left)||is_mouse_button_pressed(MouseButton::Right)||is_mouse_button_pressed(MouseButton::Middle){Some("mouse button".to_string())}
     else if mouse_wheel().1!=0.0 {Some("scroll wheel".to_string())}
     else if !managed && moved>40.0 {Some(format!("mouse moved {:.0}px",moved))}
     else {None};
    if let Some(why)=why {ss_log(&format!("exit after {:.1}s: {}",ss_age,why));break;}
   }
  }
  let raw=*audio.poll();
  if !raw.silent {mood.update(&raw,dt);}
  if is_key_pressed(KeyCode::Enter){scene_auto=!scene_auto;scene_timer=0.0;pending=None;}
  if scene_auto && pending.is_none() {
   scene_timer+=dt; // keeps counting in silence so scenes still rotate when nothing is playing
   if scene_timer>=min_dwell && (mood.boundary() || scene_timer>=max_dwell) {
    let next=if raw.silent {mood.neutral().pick(theme,&recent)} else {mood.pick(theme,&recent)};
    if next!=theme {pending=Some(next);}
    scene_timer=0.0;
   }
  }
  if let Some(p)=pending {
   fade+=dt/0.7;
   if fade>=1.0 {
    fade=1.0;recent=[theme,recent[0],recent[1]];theme=p;pending=None;mood.note_shown(theme);
    if screensaver {ss_log(&format!("scene -> {}",THEMES[theme]));}
    shockwaves.clear();scene_timer=0.0;
   }
  } else if fade>0.0 {fade=(fade-dt/1.1).max(0.);}
  if is_key_pressed(KeyCode::Escape){break;}
  if is_key_pressed(KeyCode::F12){fullscreen=!fullscreen;set_fullscreen(fullscreen);}
  if is_key_pressed(KeyCode::F11){show_details=!show_details;}
  if is_key_pressed(KeyCode::F){show_fps=!show_fps;}
  if is_key_pressed(KeyCode::G){show_grid=!show_grid;}
  if is_key_pressed(KeyCode::L){ocean_storm=!ocean_storm;}
  if is_key_pressed(KeyCode::Q){eco_mode=!eco_mode;}
  if is_key_pressed(KeyCode::Space){auto_rotate=!auto_rotate;}
  if is_key_pressed(KeyCode::Right){camera_angle+=0.12;}
  if is_key_pressed(KeyCode::Left){camera_angle-=0.12;}
  if is_key_pressed(KeyCode::Up){tilt=(tilt+3.).min(85.);}
  if is_key_pressed(KeyCode::Down){tilt=(tilt-3.).max(0.);}
  for (key,id) in [(KeyCode::Key0,0),(KeyCode::Key1,1),(KeyCode::Key2,2),(KeyCode::Key3,3),(KeyCode::Key4,4),(KeyCode::Key5,5),(KeyCode::Key6,6),(KeyCode::Key7,7),(KeyCode::Key8,8),(KeyCode::Key9,9),(KeyCode::K,10)]{
   if is_key_pressed(key){if id!=theme{recent=[theme,recent[0],recent[1]];fade=0.85;}theme=id;mood.note_shown(id);scene_timer=0.0;pending=None;}
  }
  let (_,wheel)=mouse_wheel();distance=(distance-wheel*2.).clamp(12.,130.);
  let mouse=mouse_position();
  if is_mouse_button_down(MouseButton::Left){
   camera_angle+=(mouse.0-mouse_last.0)*0.006;
   tilt=(tilt+(mouse.1-mouse_last.1)*0.15).clamp(0.,85.);
  }
  mouse_last=mouse;
  if auto_rotate {camera_angle+=dt*(0.09+0.07*raw.energy);}
  if is_key_down(KeyCode::LeftBracket){react=(react-dt*0.4).max(0.2);}
  if is_key_down(KeyCode::RightBracket){react=(react+dt*0.4).min(1.5);}
  // Scenes get a gentler version of the signal: slower attack, longer release, a small
  // kick accent on top, and an overall strength control. The analyzer's raw, fast values
  // are still used for hit detection.
  let fl=|cur:&mut f32,target:f32,att:f32,rel:f32|{let tau=if target>*cur{att}else{rel};*cur+=(target-*cur)*(1.-(-dt/tau).exp());};
  fl(&mut sb,raw.bass,0.07,0.40);fl(&mut sm,raw.mid,0.10,0.35);fl(&mut st,raw.treble,0.08,0.25);
  for i in 0..BANDS{fl(&mut spec_s[i],raw.spectrum[i],0.05,0.30);}
  let mut levels=raw;
  levels.bass=((sb*0.85+raw.kick*0.15)*react).min(1.);
  levels.mid=(sm*react).min(1.);
  levels.treble=(st*react).min(1.);
  for i in 0..BANDS{levels.spectrum[i]=(spec_s[i]*react).min(1.);}
  if is_key_down(KeyCode::Equal){audio.adjust_sensitivity(dt);}
  if is_key_down(KeyCode::Minus){audio.adjust_sensitivity(-dt);}
  let kick=levels.kick_count!=last_kick_count;
  last_kick_count=levels.kick_count;
  if kick{if flash_cool<=0.0{flash=flash.max(0.22*raw.kick);flash_cool=0.2;} if (theme==9 || theme==10) && shockwaves.len()<3 {shockwaves.push((0.0,1.0));}}
  flash_cool-=dt;
  flash=(flash-dt*1.6).max(0.);
  ocean_flash=(ocean_flash-dt*3.2).max(0.);
  thunder_timer-=dt;
  if theme==8 && ocean_storm && thunder_timer<=0.0 && (kick && raw.kick>0.8 && raw.bass>0.55) {
   ocean_flash=1.0;
   thunder_timer=3.0;
  }
  since_kick+=dt;since_snare+=dt;
  if kick{since_kick=0.;}
  if raw.snare_count!=last_snare_count{since_snare=0.;last_snare_count=raw.snare_count;}
  let frame=fx::Frame{t:time,dt,bass:levels.bass,mid:levels.mid,treble:levels.treble,air:raw.air,kick:raw.kick,snare:raw.snare,hat:raw.hihat,energy:raw.energy,beat_phase:raw.beat_phase,since_kick,since_snare,eco:eco_mode,spectrum:levels.spectrum};
  let col=palette(theme,time);
  if theme==0 {clear_background(Color::new(0.008+flash*0.05,0.028+flash*0.07,0.010+flash*0.03,1.));}
  else {clear_background(Color::new(0.005+flash*0.08,0.005+flash*0.04,0.035+flash*0.10,1.));}
  let a=camera_angle;
  let pitch=tilt.to_radians();
  let dist=distance*(1.-0.025*levels.bass);
  let (cam_pos,cam_target,cam_up,cam_fov)=if theme==1 {
   // Synthwave: a low forward drive with a slow sway; mouse drag and the arrow keys still steer it.
   let sway=camera_angle.sin();
   (vec3(sway*12.,4.5+levels.bass*0.4,44.+(distance-52.)*0.5),vec3(sway*4.,12.+(tilt-22.)*0.4,-80.),vec3(0.,1.,0.),58.)
  } else if theme==2 {
   // Beat Sync: fly down the snaking tunnel with a gentle roll.
   let (px,py)=synth::Tunnel::path(8.,time);let (tx,ty)=synth::Tunnel::path(-60.,time);
   let roll=(time*0.1).sin()*0.25+camera_angle.sin()*0.15;
   (vec3(px,py,8.),vec3(tx,ty,-60.),vec3(roll.sin(),roll.cos(),0.),70.)
  } else if theme==0 {
   // Toxic is a compact facility, so orbit closer and aim at the reactor's waist.
   let d=dist*0.62;
   (vec3(a.sin()*d*pitch.cos(),d*pitch.sin().abs()+9.0,a.cos()*d*pitch.cos()),vec3(0.,4.,0.),vec3(0.,-1.,0.),55.)
  } else {
   (vec3(a.sin()*dist*pitch.cos(),dist*pitch.sin().abs()+15.0,a.cos()*dist*pitch.cos()),vec3(0.,2.,0.),vec3(0.,-1.,0.),55.)
  };
  let camera=Camera3D{position:cam_pos,target:cam_target,up:cam_up,fovy:cam_fov,..Default::default()};
  set_camera(&camera);
  // V8 regression fix: actually render the 3D foundry when Matrix is selected.
  if theme==0 {toxic::draw(&frame);}
  if theme==3 {
   draw_rainbow_dimension(time,levels.bass,levels.mid,levels.treble,eco_mode);
   draw_rainbow_lighting(time,levels.bass,levels.mid,levels.treble,eco_mode);
   draw_prism_reactor(time,levels.bass,levels.mid,levels.treble,eco_mode);
   draw_prismatic_arcs(time,levels.bass,levels.treble,eco_mode);
   draw_kaleidoscope(time,levels.bass,eco_mode);
  }
  if theme==4 {
   draw_iron_man_lab(time,levels.bass,levels.mid,levels.treble,eco_mode);
   draw_iron_wiring(time,levels.bass,eco_mode);
  }
  if theme==6 {
   draw_matrix_foundry(time,levels.bass,levels.treble,eco_mode);
   draw_matrix_cables(time,levels.bass,eco_mode);
  }
  if theme==9 || theme==10 {
   draw_firefly_landscape(time,levels.bass,eco_mode);
   draw_firefly_grass(time,levels.bass,levels.treble,eco_mode);
   draw_firefly_ferns(time,levels.bass,levels.treble,eco_mode);
   draw_firefly_stream_details(time,levels.bass,levels.treble,eco_mode);
   draw_firefly_ancient_grove(time,levels.bass,levels.treble,eco_mode);
  }
  if theme==5 {
   draw_blizzard(time,levels.bass,levels.treble,eco_mode);
   draw_under_ice_salmon(time,levels.bass,eco_mode);
   draw_snow_clouds(time,eco_mode);
   draw_snow_moon(time,eco_mode);
   v35_snow_grey_grass(time,levels.bass,levels.mid,eco_mode);
  }
  if theme==7 {
   draw_fire_forest(time,levels.bass,eco_mode);
   v35_fire_rocks_and_reeds(time,levels.bass,eco_mode);
   draw_campfire(time,levels.bass,levels.treble,eco_mode);
  }
  if theme==8 {
   draw_ocean(time,levels.bass,levels.mid,eco_mode,if ocean_storm {ocean_flash} else {0.});
   draw_ocean_jetty(time,levels.bass,eco_mode);
   v35_ocean_shark(time,levels.bass,eco_mode);
   draw_jetty_shore(time,levels.bass,eco_mode);
   draw_seawall_spray(time,levels.bass,levels.mid,eco_mode);
   if ocean_storm {
    // Stratified 3D rainfall: evenly covers a wide area instead of clumping
    // into sine-wave bands. Each drop has stable per-cell jitter and depth.
    let side=if eco_mode {13_usize} else {20_usize};
    for ix in 0..side {
     for iz in 0..side {
      let id=(ix*73856093_usize)^(iz*19349663_usize);
      let jitter_x=((id.wrapping_mul(1664525).wrapping_add(1013904223)%1009) as f32/1009.0)-0.5;
      let jitter_z=((id.wrapping_mul(22695477).wrapping_add(1)%997) as f32/997.0)-0.5;
      let spacing=120.0/side as f32;
      let x=(ix as f32+0.5+jitter_x*0.8)*spacing-60.0;
      let z=(iz as f32+0.5+jitter_z*0.8)*spacing-60.0;
      let phase=(id%4093) as f32/4093.0;
      let fall_speed=19.0+phase*11.0+levels.treble*8.0;
      let y=(phase*52.0-time*fall_speed).rem_euclid(52.0)-8.0;
      let len=0.7+phase*1.4;
      let c=Color::new(0.34,0.56+phase*0.15,0.72,1.0);
      draw_line_3d(vec3(x,y,z),vec3(x-0.22*len,y-len,z-0.08*len),c);
      // A few droplets form expanding surface rings after impact.
      if ix%3==0 && iz%3==0 && x.abs()<43.0 && z.abs()<43.0 {
       let surface=ocean_height(x,z,time,levels.bass,levels.mid);
       let impact_age=(surface-y)/fall_speed;
       if impact_age>=0.0 && impact_age<0.70 {
        draw_ocean_splash(x,z,surface,impact_age,eco_mode);
       }
      }
     }
    }
   }
  }
  // Rebuilt scenes and per-scene detail layers (ECO versions are about a third of the size).
  if theme==1 {synth_scene.update(&frame);synth_scene.draw(&frame,show_grid);}
  if theme==2 {tunnel.update(&frame);tunnel.draw(&frame);}
  match theme {
   3=>details::rainbow(&frame),4=>details::iron(&frame),5=>details::snow(&frame),
   6=>details::matrix(&frame),7=>details::fire(&frame),8=>details::ocean(&frame),
   9=>details::firefly(&frame,false),10=>details::firefly(&frame,true),_=>{}
  }
  // Cyber Firefly V3: flocking, bass shockwaves, and visible long trails.
  if theme == 9 || theme == 10 {
   let count=if eco_mode {6} else {FIREFLY_COUNT};
   let center=fireflies.iter().take(count).fold(vec3(0.,0.,0.),|acc,f|acc+f.pos)/count as f32;
   // Small fixed-size swarm: separation prevents clumping without heap allocations.
   let mut avoidance=[Vec3::ZERO;FIREFLY_COUNT];
   for i in 0..count {
    for j in 0..count {
     if i==j {continue;}
     let delta=fireflies[i].pos-fireflies[j].pos;
     let d2=delta.length_squared();
     if d2>0.001 && d2<25. {avoidance[i]+=delta/(d2+0.5);}
    }
   }
   for (i,firefly) in fireflies.iter_mut().take(count).enumerate() {
    firefly.update(dt,time,levels.bass,levels.treble,center,avoidance[i]);
    if kick {firefly.shock(0.35+levels.bass*1.3);}
    firefly.draw(time,levels.treble,levels.bass);
   }
   for wave in &mut shockwaves {
    wave.0+=dt*(19.+levels.bass*16.);
    wave.1-=dt*0.85;
    let r=wave.0;
    let c=Color::new(0.12*wave.1.max(0.),0.95*wave.1.max(0.),0.20*wave.1.max(0.),1.);
    for i in 0..(if eco_mode {12} else {24}) {
     let segments=if eco_mode {12.} else {24.};
     let a=std::f32::consts::TAU*i as f32/segments;
     let b=std::f32::consts::TAU*(i+1) as f32/segments;
     draw_line_3d(vec3(a.cos()*r,1.4,a.sin()*r),vec3(b.cos()*r,1.4,b.sin()*r),c);
    }
   }
   shockwaves.retain(|w|w.1>0.);
  }
  // DETAIL only: slow drifting light motes give every scene some depth and life.
  if !eco_mode && theme!=1 && theme!=2 && theme!=8 && theme!=9 && theme!=10 {
   for i in 0..48u32 {
    let h=|k:u32|(i.wrapping_mul(2654435761).wrapping_add(k.wrapping_mul(97531)).wrapping_mul(2246822519)>>16) as f32/65535.;
    let tw=0.5+0.5*(time*(0.8+h(7)*1.6)+h(8)*6.283).sin();
    let p=vec3((h(1)-0.5)*90.+(time*0.07+h(4)*6.283).sin()*2.5,1.+h(3)*22.+(time*0.11+h(5)*6.283).sin()*1.2,(h(2)-0.5)*90.);
    let sz=0.05+0.07*tw+levels.treble*0.08;
    let c=Color::new(0.5+col.r*0.5,0.5+col.g*0.5,0.5+col.b*0.5,0.18+0.25*tw);
    draw_line_3d(p-vec3(sz,0.,0.),p+vec3(sz,0.,0.),c);
    draw_line_3d(p-vec3(0.,sz,0.),p+vec3(0.,sz,0.),c);
   }
  }
  set_default_camera();

  if theme==8 && ocean_storm && ocean_flash>0.05 {
   draw_rectangle(0.,0.,screen_width(),screen_height(),
       Color::new(0.65,0.82,1.0,(ocean_flash*0.12).min(0.12)));
  }
  if !eco_mode {
   let (w,h)=(screen_width(),screen_height());
   let step=h*0.03;
   for i in 0..10 {
    let c=Color::new(0.,0.,0.,0.30*(1.-i as f32/10.).powi(2));
    let o=i as f32*step;
    draw_rectangle(0.,o,w,step,c);draw_rectangle(0.,h-o-step,w,step,c);
    draw_rectangle(o,0.,step,h,c);draw_rectangle(w-o-step,0.,step,h,c);
   }
  }
  if fade>0.001 {
   let f=fade.min(1.);
   draw_rectangle(0.,0.,screen_width(),screen_height(),Color::new(0.,0.,0.,f*f*(3.-2.*f)));
  }
  if show_details {
   let tempo=if levels.tempo_conf>0.2 {format!("  |  ~{:.0} BPM",levels.bpm)} else {String::new()};
   let title=format!("BEN94Z  |  {}  |  BASS {:.2}  MID {:.2}  HIGH {:.2}",THEMES[theme],levels.bass,levels.mid,levels.treble);
   draw_text(&title,20.,35.,26.,WHITE);
   let extra=format!("REACT {:.0}%  ([ ] adjust){}{}",react*100.,tempo,if audio.is_demo() {"  |  DEMO TRACK"} else if audio.is_live() {""} else {"  |  NO AUDIO INPUT"});
   draw_text(&extra,20.,84.,20.,SKYBLUE);
   draw_text(if scene_auto {"ENTER: AUTO ON (follows the music, 30s+ per scene)"} else {"ENTER: AUTO OFF"},20.,62.,20.,GREEN);
   draw_text("0 Toxic | 3 Rainbow | 4 Iron Man | 5 Snow | 6 Matrix | 7 Fire | 8 Ocean | L Storm | 9 Fireflies | K Fireflies | Q Eco/Detail | F FPS | G Grid | ENTER Scene Auto | SPACE Auto Camera | F11 Hide HUD | F12 Fullscreen | +/- Sensitivity | [ ] Reaction | ESC Quit",20.,screen_height()-20.,17.,LIGHTGRAY);
   draw_text(if eco_mode {"Q: ECO"} else {"Q: DETAIL"},screen_width()-150.,65.,20.,YELLOW);
   if show_fps {draw_text(&format!("FPS: {}",get_fps()),screen_width()-130.,35.,23.,YELLOW);}
  }
  if !eco_mode && dt>frame_budget.as_secs_f32()*1.35 {slow_timer+=dt;} else {slow_timer=(slow_timer-dt*0.5).max(0.);}
  if slow_timer>4.0 && auto_eco {eco_mode=true;slow_timer=0.;notice=6.;}
  notice=(notice-dt).max(0.);
  if show_details && notice>0. {draw_text("Frame rate was low: switched to ECO automatically (Q for DETAIL)",20.,108.,20.,ORANGE);}
  let remaining=frame_budget.saturating_sub(frame_start.elapsed());
  if !remaining.is_zero(){std::thread::sleep(remaining);}
  next_frame().await;
 }
}

#[cfg(test)]
mod director_tests {
 use super::*;
 fn mood(bass:f32,vocal:f32,bright:f32,density:f32,perc:f32,energy:f32)->Mood{Mood{bass,vocal,bright,density,perc,energy,loud_fast:0.,loud_slow:0.,shown_age:[0;11],shown_count:[0;11]}}
 fn picks(m:&Mood,current:usize)->std::collections::HashSet<usize>{
  (0..200).map(|_|m.pick(current,&[usize::MAX;3])).collect()
 }
 #[test]
 fn heavy_bass_prefers_toxic_or_matrix(){
  let p=picks(&mood(0.9,0.1,0.2,0.5,0.5,0.7),5);
  assert!(p.iter().all(|s|*s==0||*s==6),"{p:?}");
 }
 #[test]
 fn quiet_atmosphere_prefers_snow_or_fireflies(){
  let p=picks(&mood(0.1,0.3,0.4,0.05,0.05,0.1),0);
  assert!(p.iter().all(|s|*s==5||*s==9),"{p:?}");
 }
 #[test]
 fn busy_bright_electronic_prefers_synthwave_or_beat_sync(){
  let p=picks(&mood(0.4,0.1,0.8,0.9,0.4,0.9),0);
  assert!(p.iter().all(|s|*s==1||*s==2),"{p:?}");
 }
 #[test]
 fn melodic_prefers_rainbow(){
  let p=picks(&mood(0.2,0.9,0.7,0.3,0.1,0.5),0);
  assert!(p.contains(&3),"{p:?}");
 }
 #[test]
 fn percussive_prefers_iron_man(){
  let p=picks(&mood(0.3,0.05,0.4,0.6,0.95,0.6),0);
  assert!(p.contains(&4),"{p:?}");
 }
 #[test]
 fn never_repeats_current_scene_and_avoids_recent(){
  let m=mood(0.9,0.1,0.2,0.5,0.5,0.7);
  for _ in 0..200{assert_ne!(m.pick(0,&[usize::MAX;3]),0);}
  for _ in 0..200{let n=m.pick(5,&[0,usize::MAX,usize::MAX]);assert!(n!=5&&n!=0,"{n}");}
 }
 #[test]
 fn boundary_fires_on_breakdown_and_drop_only(){
  let mut m=Mood::default();
  m.loud_slow=0.5;m.loud_fast=0.5;assert!(!m.boundary());
  m.loud_fast=0.25;assert!(m.boundary());
  m.loud_fast=0.8;assert!(m.boundary());
  m.loud_slow=0.01;m.loud_fast=0.001;assert!(!m.boundary());
 }

 fn profiles()->[(&'static str,Mood);6]{[
   ("bassy electronic",mood(0.7,0.15,0.35,0.6,0.5,0.7)),
   ("calm ambient",mood(0.2,0.3,0.4,0.1,0.1,0.2)),
   ("melodic / vocal",mood(0.3,0.7,0.6,0.3,0.15,0.5)),
   ("drum-heavy",mood(0.4,0.1,0.5,0.6,0.7,0.6)),
   ("typical pop",mood(0.4,0.4,0.5,0.4,0.35,0.5)),
   ("busy bright dance",mood(0.5,0.2,0.8,0.8,0.5,0.85)),
  ]}
 /// Simulate `n` consecutive scene changes under steady music; returns how often each scene was shown.
 fn simulate(mut m:Mood,n:usize,bonus:f32)->[u32;11]{
  let mut cur=0usize;let mut recent=[usize::MAX;3];let mut seen=[0u32;11];
  m.note_shown(cur);
  for _ in 0..n{let nx=m.pick_with(cur,&recent,bonus);recent=[cur,recent[0],recent[1]];cur=nx;m.note_shown(cur);seen[nx]+=1;}
  seen
 }
 #[test]
 fn every_scene_is_shown_for_every_kind_of_music(){
  for (name,m) in profiles(){
   let before=simulate(mood(m.bass,m.vocal,m.bright,m.density,m.perc,m.energy),40,0.0);
   let after=simulate(mood(m.bass,m.vocal,m.bright,m.density,m.perc,m.energy),40,FAIR_BONUS);
   let d=|s:&[u32;11]|s.iter().filter(|c|**c>0).count();
   println!("{:<18} distinct scenes in 40 switches: before {:>2}/11  after {:>2}/11   most shown: {}x scene {}",
    name,d(&before),d(&after),after.iter().max().unwrap(),after.iter().position(|c|c==after.iter().max().unwrap()).unwrap());
   assert_eq!(d(&after),11,"{name}: {after:?}");
   // still shaped by the music: no scene may hog more than a third of the time
   assert!(*after.iter().max().unwrap()<=14,"{name}: {after:?}");
  }
 }
 #[test]
 fn music_still_shapes_the_choice(){
  // With the fairness bonus, the best match should still be the most frequent for clear-cut music.
  let after=simulate(mood(0.9,0.05,0.2,0.5,0.4,0.7),200,FAIR_BONUS);
  let top=after.iter().position(|c|c==after.iter().max().unwrap()).unwrap();
  assert!(top==0||top==6,"heavy bass should favour Toxic/Matrix, got {top}: {after:?}");
  let calm=simulate(mood(0.1,0.3,0.4,0.05,0.05,0.1),200,FAIR_BONUS);
  let top=calm.iter().position(|c|c==calm.iter().max().unwrap()).unwrap();
  assert!(top==5||top==9,"calm music should favour Snow/Fireflies, got {top}: {calm:?}");
 }
 #[test]
 fn note_shown_tracks_waiting_time(){
  let mut m=Mood::default();m.note_shown(3);m.note_shown(5);m.note_shown(5);
  assert_eq!(m.shown_age[5],0);assert_eq!(m.shown_age[3],2);assert_eq!(m.shown_age[0],3);
 }
 #[test]
 fn silence_rotates_through_everything_without_repeats(){
  let mut m=Mood::default();let mut cur=0usize;let mut recent=[usize::MAX;3];let mut seen=[0u32;11];
  m.note_shown(cur);
  for _ in 0..33{
   let nx=m.neutral().pick(cur,&recent);
   assert_ne!(nx,cur);
   recent=[cur,recent[0],recent[1]];cur=nx;m.note_shown(cur);seen[nx]+=1;
  }
  assert!(seen.iter().all(|c|*c>=1),"{seen:?}");
  assert!(*seen.iter().max().unwrap()<=5,"one scene hogging the rotation: {seen:?}");
  println!("silent rotation over 33 switches: {seen:?}");
 }
 #[test]
 fn forest_scenes_always_get_their_turn(){
  // Long run: every forest scene (Fire 7, Firefly Forest 9, Cyber Firefly 10) keeps a real share
  // for every kind of music. Short run: a 5-scene screensaver session almost always includes one.
  for (name,m) in profiles(){
   let a=simulate(mood(m.bass,m.vocal,m.bright,m.density,m.perc,m.energy),220,FAIR_BONUS);
   for f in [7usize,9,10]{assert!(a[f]as f32/2.2>=7.0,"{name}: forest scene {f} only {:.1}%",a[f]as f32/2.2);}
   let mut with_forest=0;
   for _ in 0..300{let b=simulate(mood(m.bass,m.vocal,m.bright,m.density,m.perc,m.energy),5,FAIR_BONUS);if b[7]+b[9]+b[10]>0{with_forest+=1;}}
   assert!(with_forest>=285,"{name}: only {with_forest}/300 five-scene sessions included a forest scene");
  }
 }
}

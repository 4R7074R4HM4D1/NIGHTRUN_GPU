//! Token sampling: greedy, or temperature + top-k prefilter + top-p.

use nr_tensor::vec as tvec;

pub struct Sampler {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    rng: u64,
}

const TOP_K: usize = 64;

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler {
            temperature,
            top_p,
            rng: seed | 1,
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    // NIGHTRUN_STAGE8E_GPU_SAMPLER_V1: UEFI owns RNG and final choice.
    // GPU only returns ordered candidate IDs and unnormalized probability mass.
    // No RNG state is advanced until all protocol data has been validated.
    pub fn sample_prepared(&mut self, ids:&[u32;64], weights:&[f32;64],
                           n:usize, mass:f32) -> Option<u32> {
        if !(1..=64).contains(&n) || !mass.is_finite() || mass<=0.0 {return None;}
        let mut sum=0.0f32;
        for i in 0..n {
            if !weights[i].is_finite() || weights[i]<0.0 {return None;}
            if ids[..i].iter().any(|&id|id==ids[i]) {return None;}
            sum+=weights[i];
        }
        if (sum-mass).abs() > 0.0001*mass.max(1.0) {return None;}
        let r=self.next_f32()*mass;
        let mut acc=0.0f32;
        for i in 0..n {
            acc+=weights[i];
            if r<=acc {return Some(ids[i]);}
        }
        Some(ids[n-1])
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return tvec::argmax(logits) as u32;
        }

        // Top-k prefilter with a fixed-size min-heap keyed on logit.
        let mut heap: [(f32, u32); TOP_K] = [(f32::NEG_INFINITY, 0); TOP_K];
        for (i, &v) in logits.iter().enumerate() {
            if v > heap[0].0 {
                heap[0] = (v, i as u32);
                // Sift down.
                let mut n = 0;
                loop {
                    let l = 2 * n + 1;
                    let r = 2 * n + 2;
                    let mut small = n;
                    if l < TOP_K && heap[l].0 < heap[small].0 {
                        small = l;
                    }
                    if r < TOP_K && heap[r].0 < heap[small].0 {
                        small = r;
                    }
                    if small == n {
                        break;
                    }
                    heap.swap(n, small);
                    n = small;
                }
            }
        }

        // Softmax over the top-k with temperature.
        let mut cand: [(f32, u32); TOP_K] = heap;
        cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let max = cand[0].0;
        let mut sum = 0f32;
        let mut probs = [0f32; TOP_K];
        for (p, c) in probs.iter_mut().zip(cand.iter()) {
            *p = libm::expf((c.0 - max) / self.temperature);
            sum += *p;
        }

        // Nucleus: keep the smallest prefix with cumulative prob >= top_p.
        let cut = self.top_p * sum;
        let mut cum = 0f32;
        let mut n = TOP_K;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cut {
                n = i + 1;
                break;
            }
        }

        // Draw from the truncated distribution.
        let r = self.next_f32() * cum;
        let mut acc = 0f32;
        for i in 0..n {
            acc += probs[i];
            if r <= acc {
                return cand[i].1;
            }
        }
        cand[n - 1].1
    }
}

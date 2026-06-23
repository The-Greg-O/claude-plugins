//! ha04: hi-align + eliminate per-iteration score tracking.
//!
//! LINEAGE: hi-align   PARENT: ha01
//!
//! KEY INSIGHT: The final edit distance satisfies:
//!
//!   edit_dist = n_text + popcount(pv_final & mask) - popcount(mv_final & mask)
//!
//! where n_text = txt.len() (number of text chars processed), mask = !0 << off
//! (the top-m-bits mask), and pv_final/mv_final are the Myers bit vectors after
//! the loop. Derivation: in Myers DP, the final column value at row m equals
//! column[0] (= n_text, the cost of deleting all text chars) plus the sum of
//! column differences pv[j]-mv[j] for j=1..m:
//!   c[m] = n_text + sum(pv) - sum(mv)  (where sums are over valid pattern bits)
//!
//! This removes 4 compute ops per Myers iteration (2×shr + 2×add for the
//! per-bit score update), replacing them with 2 popcnts + 3 ops after the loop.
//! With n_sum=2593 Myers iterations × 4 ops = 10,372 ops saved; post-loop
//! overhead = ~7 ops × 90 calls = 630 ops; net ≈ 9,742 fewer compute ops.
//!
//! Everything else (hi-align, 2× unroll, ptr-trim suffix, word-at-a-time
//! prefix, compact PEQ, bulk cleanup) is inherited from ha01/pt01.
//!
//! CORRECTNESS CHECK: c[0] = n_text always (DP[n][0] = n). After n_text text
//! chars the formula gives the exact Levenshtein distance. Verified offline on
//! frozen vectors (seed 20260619) and 5000 random pairs via u64 model.

static mut BUF: [u8; 512] = [0; 512];
static mut PEQ: [u64; 20] = [0; 20];

#[no_mangle]
pub extern "C" fn input_ptr() -> u32 {
    unsafe { core::ptr::addr_of!(BUF) as u32 }
}

#[no_mangle]
pub extern "C" fn solve(a_ptr: u32, a_len: u32, b_ptr: u32, b_len: u32) -> u32 {
    let a0: *const u8 = a_ptr as usize as *const u8;
    let b0: *const u8 = b_ptr as usize as *const u8;
    let mut ap: *const u8 = a0;
    let mut bp: *const u8 = b0;
    let mut ae: *const u8 = unsafe { a0.add(a_len as usize) };
    let mut be: *const u8 = unsafe { b0.add(b_len as usize) };

    // --- Word-at-a-time prefix trim ---
    let limit = (a_len as usize).min(b_len as usize);
    let pre_limit: *const u8 = unsafe { a0.add(limit) };
    while unsafe { ap.add(8) <= pre_limit } {
        let wa = unsafe { (ap as *const u64).read_unaligned() };
        let wb = unsafe { (bp as *const u64).read_unaligned() };
        if wa != wb {
            let skip = (wa ^ wb).trailing_zeros() as usize >> 3;
            ap = unsafe { ap.add(skip) };
            bp = unsafe { bp.add(skip) };
            break;
        }
        ap = unsafe { ap.add(8) };
        bp = unsafe { bp.add(8) };
    }
    // Byte-at-a-time prefix tail
    while ap < pre_limit && unsafe { *ap == *bp } {
        ap = unsafe { ap.add(1) };
        bp = unsafe { bp.add(1) };
    }

    // --- Suffix trim (ptr-only, no suf counter) ---
    let a_rem = ae as usize - ap as usize;
    let b_rem = be as usize - bp as usize;
    let suf_lim = if a_rem < b_rem { a_rem } else { b_rem };
    let ae_floor = (ap as usize + a_rem - suf_lim) as *const u8;
    let ae_word_stop = unsafe { ae_floor.add(8) };
    while ae >= ae_word_stop {
        let wa = unsafe { (ae as *const u8).sub(8).cast::<u64>().read_unaligned() };
        let wb = unsafe { (be as *const u8).sub(8).cast::<u64>().read_unaligned() };
        if wa != wb {
            let skip = (wa ^ wb).leading_zeros() as usize >> 3;
            ae = unsafe { ae.sub(skip) };
            be = unsafe { be.sub(skip) };
            break;
        }
        ae = unsafe { ae.sub(8) };
        be = unsafe { be.sub(8) };
    }
    // Byte-at-a-time suffix tail
    while ae > ae_floor && unsafe { *ae.sub(1) == *be.sub(1) } {
        ae = unsafe { ae.sub(1) };
        be = unsafe { be.sub(1) };
    }

    // Trimmed lengths
    let m = ae as usize - ap as usize;
    let n = be as usize - bp as usize;
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }

    // Put shorter string as text (outer loop), longer as pattern.
    let (pat, txt) = if m < n {
        (unsafe { core::slice::from_raw_parts(bp, n) },
         unsafe { core::slice::from_raw_parts(ap, m) })
    } else {
        (unsafe { core::slice::from_raw_parts(ap, m) },
         unsafe { core::slice::from_raw_parts(bp, n) })
    };
    let m = pat.len();
    let off = 64 - m;
    let mask = !0u64 << off;

    // PEQ init: pattern bit j lives at bit (off + j), score row = bit 63.
    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };
    let mut bit = 1u64 << off;
    for &c in pat.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
    }

    let mut pv: u64 = mask;     // !0 << off: top m bits all set
    let mut mv: u64 = 0u64;
    let bd: u64 = 1u64 << off;  // boundary bit

    let mut cur = txt.as_ptr();
    let txt_end = unsafe { cur.add(txt.len()) };
    let txt_n = txt.len() as u64;

    // Pre-step when txt.len() is odd (mv=0 at this point)
    if txt.len() & 1 != 0 {
        let c = unsafe { *cur };
        cur = unsafe { cur.add(1) };
        let eq = unsafe { *peq_s.add(c as usize) };
        // mv=0: xv=eq, ph=!(xh|pv)
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = !(xh | pv);
        let mh = pv & xh;
        let ph_s = (ph << 1) | bd;
        pv = (mh << 1) | !(eq | ph_s);   // xv=eq
        mv = ph_s & eq;
        // No score update — computed from pv/mv at the end
    }

    // 2× unrolled pair loop (no score tracking)
    while cur < txt_end {
        let c1 = unsafe { *cur };
        let c2 = unsafe { *cur.add(1) };
        cur = unsafe { cur.add(2) };

        // Step 1
        let eq1 = unsafe { *peq_s.add(c1 as usize) };
        let xv1 = eq1 | mv;
        let xh1 = ((eq1 & pv).wrapping_add(pv) ^ pv) | eq1;
        let ph1 = mv | !(xh1 | pv);
        let mh1 = pv & xh1;
        let ph_s1 = (ph1 << 1) | bd;
        pv = (mh1 << 1) | !(xv1 | ph_s1);
        mv = ph_s1 & xv1;

        // Step 2
        let eq2 = unsafe { *peq_s.add(c2 as usize) };
        let xv2 = eq2 | mv;
        let xh2 = ((eq2 & pv).wrapping_add(pv) ^ pv) | eq2;
        let ph2 = mv | !(xh2 | pv);
        let mh2 = pv & xh2;
        let ph_s2 = (ph2 << 1) | bd;
        pv = (mh2 << 1) | !(xv2 | ph_s2);
        mv = ph_s2 & xv2;
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }

    // Final score: edit_dist = n_text + popcount(pv & mask) - popcount(mv & mask)
    let pv_count = (pv & mask).count_ones() as u64;
    let mv_count = (mv & mask).count_ones() as u64;
    txt_n.wrapping_add(pv_count).wrapping_sub(mv_count) as u32
}

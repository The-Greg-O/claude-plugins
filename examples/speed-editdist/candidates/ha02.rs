//! ha02: ha01 + 4× Myers outer loop unroll.
//!
//! Lineage: hi-align. Parent: ha01.
//!
//! ha01 confirmed n_avg≈28.8 (measured, not inferred). With n_sum=2593 Myers
//! iterations across 90 cases, each loop-overhead op saved ≈ 2,593 fuel.
//!
//! The 2× unroll (inherited from cpq05) has loop overhead ~2.5 ops/char
//! (2 loads + advance-2 + compare + branch per pair). 4× overhead is ~1.75
//! ops/char (4 loads + advance-4 + compare + branch per quad). Saving ≈
//! 0.75 ops/char × 2593 ≈ 1,945 fuel expected; LLVM efficiency ~78% from
//! the 1×→2× datapoint (1,018/1,297) gives ~1,500 fuel estimate.
//!
//! Pre-step structure: handle `n & 2` first (pair), then `n & 1` (single),
//! so the main quad loop runs over the remaining `n & !3` chars — always a
//! multiple of 4. This mirrors how the 2× loop handled `n & 1` parity.

static mut BUF: [u8; 512] = [0; 512];
static mut PEQ: [u64; 20] = [0; 20];

#[no_mangle]
pub extern "C" fn input_ptr() -> u32 {
    unsafe { core::ptr::addr_of!(BUF) as u32 }
}

#[inline(always)]
unsafe fn myers_step(
    c: u8,
    peq_s: *const u64,
    pv: &mut u64,
    mv: &mut u64,
    score: &mut u64,
    bd: u64,
) {
    let eq = *peq_s.add(c as usize);
    let xv = eq | *mv;
    let xh = ((eq & *pv).wrapping_add(*pv) ^ *pv) | eq;
    let ph = *mv | !(xh | *pv);
    let mh = *pv & xh;
    *score = score.wrapping_add(ph >> 63).wrapping_sub(mh >> 63);
    let ph_s = (ph << 1) | bd;
    *pv = (mh << 1) | !(xv | ph_s);
    *mv = ph_s & xv;
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

    // --- Suffix trim using only end-pointers (no `suf` counter) ---
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

    // Put shorter string as text (outer loop). Pattern = longer string.
    let (pat, txt) = if m < n {
        (unsafe { core::slice::from_raw_parts(bp, n) },
         unsafe { core::slice::from_raw_parts(ap, m) })
    } else {
        (unsafe { core::slice::from_raw_parts(ap, m) },
         unsafe { core::slice::from_raw_parts(bp, n) })
    };
    let m = pat.len();
    let off = 64 - m; // m in 1..=64 -> off in 0..=63

    // PEQ init: pattern bit j lives at bit (off + j), so the last row is bit 63.
    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };
    let mut bit = 1u64 << off;
    for &c in pat.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
    }

    let mut pv: u64 = !0u64 << off;
    let mut mv: u64 = 0u64;
    let mut score: u64 = m as u64;
    let bd: u64 = 1u64 << off;

    let n_txt = txt.len();
    let mut cur = txt.as_ptr();
    let txt_end = unsafe { cur.add(n_txt) };

    // Pre-steps to align to a multiple of 4 for the quad loop.
    if n_txt & 2 != 0 {
        let c1 = unsafe { *cur };
        let c2 = unsafe { *cur.add(1) };
        cur = unsafe { cur.add(2) };
        unsafe {
            myers_step(c1, peq_s, &mut pv, &mut mv, &mut score, bd);
            myers_step(c2, peq_s, &mut pv, &mut mv, &mut score, bd);
        }
    }
    if n_txt & 1 != 0 {
        let c = unsafe { *cur };
        cur = unsafe { cur.add(1) };
        unsafe { myers_step(c, peq_s, &mut pv, &mut mv, &mut score, bd); }
    }
    // Quad loop: n_txt & !3 remaining chars, guaranteed divisible by 4.
    while cur < txt_end {
        let c1 = unsafe { *cur };
        let c2 = unsafe { *cur.add(1) };
        let c3 = unsafe { *cur.add(2) };
        let c4 = unsafe { *cur.add(3) };
        cur = unsafe { cur.add(4) };
        unsafe {
            myers_step(c1, peq_s, &mut pv, &mut mv, &mut score, bd);
            myers_step(c2, peq_s, &mut pv, &mut mv, &mut score, bd);
            myers_step(c3, peq_s, &mut pv, &mut mv, &mut score, bd);
            myers_step(c4, peq_s, &mut pv, &mut mv, &mut score, bd);
        }
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }
    score as u32
}

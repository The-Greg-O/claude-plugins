//! ha06: hi-align + precomputed text EQ table (TEQ).
//!
//! LINEAGE: hi-align   PARENT: ha04
//!
//! KEY INSIGHT: In ha04's Myers inner loop each text character requires:
//!   (1) i32.load8_u to fetch the byte,
//!   (2) i32.const 3 + i32.shl to convert to byte offset (c * 8),
//!   (3) i64.load to fetch PEQ[c].
//!
//! That is ~3-4 wasmi IR ops per character in the hot loop (×2 per pair
//! × ~1296 pairs over 90 cases). Precomputing TEQ[i] = PEQ[txt[i]] in a
//! one-pass loop before Myers replaces those 3-4 ops/char with a single
//! sequential i64.load per character in the inner loop — a simpler address
//! computation that wasmi / LLVM can fold more aggressively.
//!
//! Overhead: ~7 ops × 28.8 iters precompute + TEQ cleanup (txt.len() u64s
//! via write_bytes). Estimated net saving: 2,000-6,000 fuel.
//!
//! Everything else (hi-align, 2× unroll, ptr-trim suffix, word-at-a-time
//! prefix, compact PEQ, bulk cleanup, post-loop popcount) is inherited from ha04.

static mut BUF: [u8; 512] = [0; 512];
static mut PEQ: [u64; 20] = [0; 20];
static mut TEQ: [u64; 64] = [0; 64];

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

    // Precompute TEQ: for each text char, look up its PEQ value once.
    let txt_n = txt.len();
    let teq = unsafe { TEQ.as_mut_ptr() };
    {
        let mut rp = txt.as_ptr();
        let rp_end = unsafe { rp.add(txt_n) };
        let mut wp = teq;
        while rp < rp_end {
            unsafe {
                *wp = *peq_s.add(*rp as usize);
                rp = rp.add(1);
                wp = wp.add(1);
            }
        }
    }

    let mut pv: u64 = mask;
    let mut mv: u64 = 0u64;
    let bd: u64 = 1u64 << off;

    let mut teq_cur = teq as *const u64;
    let teq_end = unsafe { teq_cur.add(txt_n) };

    // Pre-step when txt.len() is odd (mv=0 at this point)
    if txt_n & 1 != 0 {
        let eq = unsafe { *teq_cur };
        teq_cur = unsafe { teq_cur.add(1) };
        // mv=0: xv=eq, ph=!(xh|pv)
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = !(xh | pv);
        let mh = pv & xh;
        let ph_s = (ph << 1) | bd;
        pv = (mh << 1) | !(eq | ph_s);   // xv=eq
        mv = ph_s & eq;
    }

    // 2× unrolled pair loop: sequential reads from TEQ (no char load, no shift-multiply)
    while teq_cur < teq_end {
        let eq1 = unsafe { *teq_cur };
        let eq2 = unsafe { *teq_cur.add(1) };
        teq_cur = unsafe { teq_cur.add(2) };

        // Step 1
        let xv1 = eq1 | mv;
        let xh1 = ((eq1 & pv).wrapping_add(pv) ^ pv) | eq1;
        let ph1 = mv | !(xh1 | pv);
        let mh1 = pv & xh1;
        let ph_s1 = (ph1 << 1) | bd;
        pv = (mh1 << 1) | !(xv1 | ph_s1);
        mv = ph_s1 & xv1;

        // Step 2
        let xv2 = eq2 | mv;
        let xh2 = ((eq2 & pv).wrapping_add(pv) ^ pv) | eq2;
        let ph2 = mv | !(xh2 | pv);
        let mh2 = pv & xh2;
        let ph_s2 = (ph2 << 1) | bd;
        pv = (mh2 << 1) | !(xv2 | ph_s2);
        mv = ph_s2 & xv2;
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }
    unsafe { core::ptr::write_bytes(teq, 0, txt_n); }

    // Final score: edit_dist = n_text + popcount(pv & mask) - popcount(mv & mask)
    let pv_count = (pv & mask).count_ones() as u64;
    let mv_count = (mv & mask).count_ones() as u64;
    (txt_n as u64).wrapping_add(pv_count).wrapping_sub(mv_count) as u32
}

//! ha05: hi-align + 3× Myers unroll.
//!
//! LINEAGE: hi-align   PARENT: ha04
//!
//! KEY INSIGHT: 3× unroll reduces per-char loop overhead vs 2× unroll.
//!
//! In wasmi's register-based IR, local.get/set/tee and i32.const/i64.const
//! are zero-fuel (register copies). Only arithmetic, memory, and branch ops
//! cost 1 fuel each. Under this model:
//!
//!   2× pair loop: step1(18) + step2(24 = 18+3_extra+3_ctrl) = 42 ops / 2 chars = 21/char
//!   3× trip loop: step1(18) + step2(19) + step3(20) + ctrl(3) = 60 ops / 3 chars = 20/char
//!
//! step2 extra vs step1: 1 op for lazy mv1 = ph_s1 & xv1.
//! step3 extra vs step1: 1 op for lazy mv2 = ph_s2 & xv2, 1 op to save mv3.
//! loop ctrl: 3 ops (add cur+3, compare, branch).
//!
//! Savings: 1 op/char × ~2,503 main-loop chars = ~2,503 fuel.
//! Extra pre-step overhead for n%3 alignment: ~810 fuel.
//! Net estimated: ~1,693 fuel (~1.9% cut from 88,976).
//!
//! Pre-steps: compute n%3 (1 op), then process 0/1/2 chars to align.
//! First pre-step always uses mv=0 optimization (saves 2 ops).
//!
//! Everything else inherited from ha04.

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

    // --- Suffix trim (ptr-only) ---
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
    while ae > ae_floor && unsafe { *ae.sub(1) == *be.sub(1) } {
        ae = unsafe { ae.sub(1) };
        be = unsafe { be.sub(1) };
    }

    let m = ae as usize - ap as usize;
    let n = be as usize - bp as usize;
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }

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
    let bd: u64 = 1u64 << off;

    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };
    let mut bit = 1u64 << off;
    for &c in pat.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
    }

    let mut pv: u64 = mask;
    let mut mv: u64 = 0u64;

    let mut cur = txt.as_ptr();
    let txt_end = unsafe { cur.add(txt.len()) };
    let txt_n = txt.len() as u64;

    // --- n%3 pre-steps to align to triple boundary ---
    let rem = txt.len() % 3;
    if rem != 0 {
        // Pre-step A: always with mv=0 (initial state)
        let c = unsafe { *cur };
        cur = unsafe { cur.add(1) };
        let eq = unsafe { *peq_s.add(c as usize) };
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = !(xh | pv);           // mv=0 optimization: no | mv
        let mh = pv & xh;
        let ph_s = (ph << 1) | bd;
        pv = (mh << 1) | !(eq | ph_s); // xv=eq when mv=0
        mv = ph_s & eq;

        if rem == 2 {
            // Pre-step B: general (mv from step A)
            let c = unsafe { *cur };
            cur = unsafe { cur.add(1) };
            let eq = unsafe { *peq_s.add(c as usize) };
            let xv = eq | mv;
            let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
            let ph = mv | !(xh | pv);
            let mh = pv & xh;
            let ph_s = (ph << 1) | bd;
            pv = (mh << 1) | !(xv | ph_s);
            mv = ph_s & xv;
        }
    }

    // --- 3× unrolled Myers loop (no score tracking) ---
    while cur < txt_end {
        let c1 = unsafe { *cur };
        let c2 = unsafe { *cur.add(1) };
        let c3 = unsafe { *cur.add(2) };
        cur = unsafe { cur.add(3) };

        // Step 1 (uses mv from previous iteration)
        let eq1 = unsafe { *peq_s.add(c1 as usize) };
        let xv1 = eq1 | mv;
        let xh1 = ((eq1 & pv).wrapping_add(pv) ^ pv) | eq1;
        let ph1 = mv | !(xh1 | pv);
        let mh1 = pv & xh1;
        let ph_s1 = (ph1 << 1) | bd;
        pv = (mh1 << 1) | !(xv1 | ph_s1);
        // mv1 = ph_s1 & xv1 — computed lazily in step 2

        // Step 2 (lazy mv1)
        let eq2 = unsafe { *peq_s.add(c2 as usize) };
        let mv1 = ph_s1 & xv1;
        let xv2 = eq2 | mv1;
        let xh2 = ((eq2 & pv).wrapping_add(pv) ^ pv) | eq2;
        let ph2 = mv1 | !(xh2 | pv);
        let mh2 = pv & xh2;
        let ph_s2 = (ph2 << 1) | bd;
        pv = (mh2 << 1) | !(xv2 | ph_s2);
        // mv2 = ph_s2 & xv2 — computed lazily in step 3

        // Step 3 (lazy mv2)
        let eq3 = unsafe { *peq_s.add(c3 as usize) };
        let mv2 = ph_s2 & xv2;
        let xv3 = eq3 | mv2;
        let xh3 = ((eq3 & pv).wrapping_add(pv) ^ pv) | eq3;
        let ph3 = mv2 | !(xh3 | pv);
        let mh3 = pv & xh3;
        let ph_s3 = (ph3 << 1) | bd;
        pv = (mh3 << 1) | !(xv3 | ph_s3);
        mv = ph_s3 & xv3;  // save mv3 for next triple
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }

    let pv_count = (pv & mask).count_ones() as u64;
    let mv_count = (mv & mask).count_ones() as u64;
    txt_n.wrapping_add(pv_count).wrapping_sub(mv_count) as u32
}

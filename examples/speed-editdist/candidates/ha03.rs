//! ha03: ha01 + base-pointer PEQ init to hint LLVM toward i32.load8_u offset=n.
//!
//! Lineage: hi-align. Parent: ha01.
//!
//! ha02 showed 4x Myers unroll HURT (+1,408). LLVM already optimizes the 2x
//! pair loop well. This iteration targets PEQ init instead.
//!
//! Problem: LLVM's 4x-unrolled PEQ init uses
//!   `local.get 4; i32.const n; i32.add; i32.load8_u` for chars 2,3,4
//! — 2 wasmi IR ops each (AddImm + Load8U). If we provide an explicit
//! base ptr for each 4-group, LLVM can emit `i32.load8_u offset=n`
//! (base-relative load with immediate offset = 1 wasmi IR op). This
//! saves 1 op per char for chars 2,3,4 in each 4-group.
//!
//! With m_sum=2701 chars across 90 cases:
//!   3 chars/group × 675 groups = 2,025 saved wasmi IR instructions ≈ 2,025 fuel.
//!
//! Key: declare `let base = pat.as_ptr().add(i)` once per 4-group, then
//! load all 4 chars as `*base`, `*base.add(1)`, `*base.add(2)`, `*base.add(3)`.
//! LLVM should recognize the shared base and emit memory-immediate loads.

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
    while ap < pre_limit && unsafe { *ap == *bp } {
        ap = unsafe { ap.add(1) };
        bp = unsafe { bp.add(1) };
    }

    // --- Suffix trim using only end-pointers ---
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

    // PEQ init: use explicit base pointer per 4-group so LLVM emits
    // i32.load8_u offset=n (1 wasmi IR op) instead of add+load8 (2 ops).
    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };
    let mut bit = 1u64 << off;
    let pat_base = pat.as_ptr();
    let mut pi = 0usize;

    // Handle m % 4 remainder first (0-3 chars)
    let pre = m & 3;
    while pi < pre {
        let c = unsafe { *pat_base.add(pi) };
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
        pi += 1;
    }
    // 4× unrolled main init loop with shared base pointer per group
    while pi < m {
        let base = unsafe { pat_base.add(pi) };
        let c1 = unsafe { *base };
        let c2 = unsafe { *base.add(1) };
        let c3 = unsafe { *base.add(2) };
        let c4 = unsafe { *base.add(3) };
        unsafe { *peq_s.add(c1 as usize) |= bit; }
        bit = bit.wrapping_shl(1);
        unsafe { *peq_s.add(c2 as usize) |= bit; }
        bit = bit.wrapping_shl(1);
        unsafe { *peq_s.add(c3 as usize) |= bit; }
        bit = bit.wrapping_shl(1);
        unsafe { *peq_s.add(c4 as usize) |= bit; }
        bit = bit.wrapping_shl(1);
        pi += 4;
    }

    let mut pv: u64 = !0u64 << off;
    let mut mv: u64 = 0u64;
    let mut score: u64 = m as u64;
    let bd: u64 = 1u64 << off;

    let mut cur = txt.as_ptr();
    let txt_end = unsafe { cur.add(txt.len()) };

    // 2× unrolled Myers outer loop (unchanged from ha01)
    if txt.len() & 1 != 0 {
        let c = unsafe { *cur };
        cur = unsafe { cur.add(1) };
        unsafe { myers_step(c, peq_s, &mut pv, &mut mv, &mut score, bd); }
    }
    while cur < txt_end {
        let c1 = unsafe { *cur };
        let c2 = unsafe { *cur.add(1) };
        cur = unsafe { cur.add(2) };
        unsafe {
            myers_step(c1, peq_s, &mut pv, &mut mv, &mut score, bd);
            myers_step(c2, peq_s, &mut pv, &mut mv, &mut score, bd);
        }
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }
    score as u32
}

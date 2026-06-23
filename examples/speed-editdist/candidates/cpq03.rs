//! cpq03: cpq01 with reversed PEQ bit order → cheaper score extraction.
//!
//! cpq01 extracts bit (m-1) for score update: `ph >> m1 & 1`, `mh >> m1 & 1`.
//! In wasm this costs 6 instructions each (get ph, get m1, shr_u, wrap, const 1, and).
//! 12 instructions total for the two extractions, plus the m1 local is live throughout.
//!
//! Insight: reverse the PEQ bit order (bit i → bit m-1-i).
//! The exit bit becomes bit 0 in the reversed vector, so extraction costs 4 instructions
//! each (get ph, const 1, and, wrap) = 8 total. Saves 4 per iteration.
//! Also eliminates the m1 local from the inner loop. Net: ~5 instrs/iter saved.
//!
//! Correctness: score tracks dp[m][j] = dp[m][j-1] + ph_bit(m-1) - mh_bit(m-1).
//! With reversed bits: ph_reversed[0] = ph_standard[m-1]. Update is identical. ✓
//!
//! Changes vs cpq01:
//!   PEQ init: bit starts at 1<<(m-1) and shifts RIGHT (reversed bit assignment).
//!   score update: `ph & 1 != 0` and `mh & 1 != 0` (extract bit 0, no shift needed).
//!   ph_s: `(ph >> 1) | high` instead of `(ph << 1) | 1` (boundary at bit m-1).
//!   pv update: `(mh >> 1) | !(xv | ph_s)` instead of `(mh << 1) | ...`.
//!   high: still used for ph_s boundary, no longer needed in score update.
//!   m1 variable: eliminated.

static mut BUF: [u8; 512] = [0; 512];
static mut PEQ: [u64; 20] = [0; 20];

#[no_mangle]
pub extern "C" fn input_ptr() -> u32 {
    unsafe { core::ptr::addr_of!(BUF) as u32 }
}

#[no_mangle]
pub extern "C" fn solve(a_ptr: u32, a_len: u32, b_ptr: u32, b_len: u32) -> u32 {
    let mut a = unsafe { core::slice::from_raw_parts(a_ptr as *const u8, a_len as usize) };
    let mut b = unsafe { core::slice::from_raw_parts(b_ptr as *const u8, b_len as usize) };

    // Word-at-a-time prefix trim.
    let limit = a.len().min(b.len());
    let mut pre = 0usize;
    while pre + 8 <= limit {
        let wa = unsafe { (a.as_ptr().add(pre) as *const u64).read_unaligned() };
        let wb = unsafe { (b.as_ptr().add(pre) as *const u64).read_unaligned() };
        if wa != wb {
            pre += ((wa ^ wb).trailing_zeros() >> 3) as usize;
            break;
        }
        pre += 8;
    }
    while pre < limit {
        if unsafe { *a.get_unchecked(pre) != *b.get_unchecked(pre) } { break; }
        pre += 1;
    }
    a = &a[pre..];
    b = &b[pre..];

    // Word-at-a-time suffix trim.
    let limit = a.len().min(b.len());
    let mut suf = 0usize;
    while suf + 8 <= limit {
        let wa = unsafe { (a.as_ptr().add(a.len() - suf - 8) as *const u64).read_unaligned() };
        let wb = unsafe { (b.as_ptr().add(b.len() - suf - 8) as *const u64).read_unaligned() };
        if wa != wb {
            suf += ((wa ^ wb).leading_zeros() >> 3) as usize;
            break;
        }
        suf += 8;
    }
    while suf < limit {
        if unsafe {
            *a.get_unchecked(a.len() - 1 - suf) != *b.get_unchecked(b.len() - 1 - suf)
        } { break; }
        suf += 1;
    }
    a = &a[..a.len() - suf];
    b = &b[..b.len() - suf];

    let m = a.len();
    let n = b.len();
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }

    let (a, b) = if m < n { (b, a) } else { (a, b) };
    let m = a.len();

    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };

    // PEQ init with REVERSED bit order: position i in pattern → bit (m-1-i).
    // Equivalent to reversing both strings (edit_dist is symmetric under reversal).
    let high: u64 = if m < 64 { 1u64 << (m - 1) } else { 1u64 << 63 };
    let mut bit = high;
    for &c in a.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shr(1);
    }

    // pv: bits 0..m-1 all set — same as standard (bit reversal is within the active range).
    let mut pv: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut mv: u64 = 0u64;
    let mut score = m as u32;
    // `high` is the boundary mask for ph_s (bit m-1 = the "start" boundary in reversed order).

    for &c in b.iter() {
        let eq = unsafe { *peq_s.add(c as usize) };
        let xv = eq | mv;
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;
        // Score update: extract bit 0 (4 wasm instrs each vs 6 in cpq01).
        score = score.wrapping_add(
            ((ph & 1 != 0) as i32 - (mh & 1 != 0) as i32) as u32,
        );
        // Boundary at bit m-1 in reversed ordering.
        let ph_s = (ph >> 1) | high;
        pv = (mh >> 1) | !(xv | ph_s);
        mv = ph_s & xv;
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }
    score
}

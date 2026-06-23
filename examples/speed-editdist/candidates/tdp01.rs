//! tdp01: cpq05 + specialized n==1 path after trim/swap.
//!
//! After trim + role-swap (pattern=longer, text=shorter), if text has exactly
//! 1 character, skip Myers entirely: scan pattern a for text char c.
//! edit_dist(a, [c]) = m-1 if c in a, else m.
//! Cost: ~12*k ops (k=chars scanned, early-exit) vs Myers ~12*m+89 ops.

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
    m_shift: u64,
) {
    let eq = *peq_s.add(c as usize);
    let xv = eq | *mv;
    let xh = ((eq & *pv).wrapping_add(*pv) ^ *pv) | eq;
    let ph = *mv | !(xh | *pv);
    let mh = *pv & xh;
    *score = score
        .wrapping_add((ph >> m_shift) & 1)
        .wrapping_sub((mh >> m_shift) & 1);
    let ph_s = (ph << 1) | 1;
    *pv = (mh << 1) | !(xv | ph_s);
    *mv = ph_s & xv;
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

    // Shorter string as text (fewer outer-loop iterations).
    let (a, b) = if m < n { (b, a) } else { (a, b) };
    let m = a.len();
    let n = b.len();

    // Specialized path for n == 1: scan pattern for the single text character.
    // edit_dist(a[0..m], [c]) = m - 1 if c appears anywhere in a, else m.
    if n == 1 {
        let c = unsafe { *b.as_ptr() };
        let mut p = a.as_ptr();
        let end = unsafe { p.add(m) };
        while p < end {
            if unsafe { *p } == c { return (m - 1) as u32; }
            p = unsafe { p.add(1) };
        }
        return m as u32;
    }

    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };

    // PEQ init.
    let mut bit = 1u64;
    for &c in a.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
    }

    let mut pv: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut mv: u64 = 0u64;
    let mut score: u64 = m as u64;
    let m_shift = (m - 1) as u64;

    let mut bp = b.as_ptr();
    let b_end = unsafe { bp.add(n) };

    // If n is odd, process one character first so the pair loop gets an even count.
    if n & 1 != 0 {
        let c = unsafe { *bp };
        bp = unsafe { bp.add(1) };
        unsafe { myers_step(c, peq_s, &mut pv, &mut mv, &mut score, m_shift); }
    }

    // Process remaining characters in pairs (2x unrolled Myers outer loop).
    while bp < b_end {
        let c1 = unsafe { *bp };
        let c2 = unsafe { *bp.add(1) };
        bp = unsafe { bp.add(2) };
        unsafe {
            myers_step(c1, peq_s, &mut pv, &mut mv, &mut score, m_shift);
            myers_step(c2, peq_s, &mut pv, &mut mv, &mut score, m_shift);
        }
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }

    score as u32
}

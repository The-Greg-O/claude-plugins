//! cpq04: cpq01 + u64 score to eliminate 2 i32.wrap_i64 per Myers iteration.
//!
//! cpq01 inner loop (71 instrs) extracts ph/mh high-bits via:
//!   i64.shr_u → i32.wrap_i64 → i32.const 1 → i32.and  (4 ops each, ×2 = 8)
//! With score as u64, LLVM stays in i64 throughout, emitting:
//!   i64.shr_u → i64.const 1 → i64.and                  (3 ops each, ×2 = 6)
//! Saves 2 instructions per Myers outer-loop iteration, expected ~3,000 fuel.

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

    // Shorter string as text (fewer outer-loop iterations).
    let (a, b) = if m < n { (b, a) } else { (a, b) };
    let m = a.len();

    let peq_s = unsafe { PEQ.as_mut_ptr().wrapping_sub(b'a' as usize) };

    // PEQ init.
    let mut bit = 1u64;
    for &c in a.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
    }

    let mut pv: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut mv: u64 = 0u64;
    // u64 score: keeps score in i64 registers throughout the loop,
    // eliminating i32.wrap_i64 at both bit-extraction sites.
    let mut score: u64 = m as u64;
    let m_shift = (m - 1) as u64;

    for &c in b.iter() {
        let eq = unsafe { *peq_s.add(c as usize) };
        let xv = eq | mv;
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;
        // Both extractions stay i64 — no i32.wrap_i64 instruction emitted.
        score = score
            .wrapping_add((ph >> m_shift) & 1)
            .wrapping_sub((mh >> m_shift) & 1);
        let ph_s = (ph << 1) | 1;
        pv = (mh << 1) | !(xv | ph_s);
        mv = ph_s & xv;
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }

    score as u32
}

//! Myers/Hyyrö bit-parallel + word-at-a-time prefix/suffix trim (hybrid02).
//! Upgrades hybrid01's byte-at-a-time trim to u64 word scan: 8 bytes per
//! comparison; trailing/leading-zeros locate the first mismatch byte.

static mut BUF: [u8; 512] = [0; 512];
static mut PEQ: [u64; 256] = [0; 256];

#[no_mangle]
pub extern "C" fn input_ptr() -> u32 {
    unsafe { core::ptr::addr_of!(BUF) as u32 }
}

#[no_mangle]
pub extern "C" fn solve(a_ptr: u32, a_len: u32, b_ptr: u32, b_len: u32) -> u32 {
    let mut a = unsafe { core::slice::from_raw_parts(a_ptr as *const u8, a_len as usize) };
    let mut b = unsafe { core::slice::from_raw_parts(b_ptr as *const u8, b_len as usize) };

    // Word-at-a-time prefix trim: compare 8 bytes per iteration.
    let limit = a.len().min(b.len());
    let mut pre = 0usize;
    while pre + 8 <= limit {
        let wa = unsafe { (a.as_ptr().add(pre) as *const u64).read_unaligned() };
        let wb = unsafe { (b.as_ptr().add(pre) as *const u64).read_unaligned() };
        if wa != wb {
            // Little-endian: LSB = byte at lowest address.
            // trailing_zeros/8 = index of first differing byte within this word.
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
        // Load 8 bytes ending at the current suffix boundary (going backwards).
        let wa = unsafe { (a.as_ptr().add(a.len() - suf - 8) as *const u64).read_unaligned() };
        let wb = unsafe { (b.as_ptr().add(b.len() - suf - 8) as *const u64).read_unaligned() };
        if wa != wb {
            // Little-endian: last byte in memory = MSB of u64.
            // leading_zeros/8 = matching bytes counted from the string end.
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

    // Myers with shorter string as text (fewer outer iterations).
    let (a, b) = if m < n { (b, a) } else { (a, b) };
    let m = a.len();

    for (i, &c) in a.iter().enumerate() {
        unsafe { *PEQ.get_unchecked_mut(c as usize) |= 1u64 << i; }
    }

    let mut pv: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut mv: u64 = 0u64;
    let mut score = m as u32;
    let high = 1u64 << (m - 1);

    for &c in b.iter() {
        let eq = unsafe { *PEQ.get_unchecked(c as usize) };
        let xv = eq | mv;
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;
        score += (ph & high != 0) as u32;
        score -= (mh & high != 0) as u32;
        let ph_s = (ph << 1) | 1;
        pv = (mh << 1) | !(xv | ph_s);
        mv = ph_s & xv;
    }

    for &c in a.iter() {
        unsafe { *PEQ.get_unchecked_mut(c as usize) = 0; }
    }

    score
}

//! Myers/Hyyrö bit-parallel + prefix/suffix trimming (hybrid lineage).
//! Common prefix/suffix is stripped before Myers so the inner loop only
//! processes the "edit region". Shorter trimmed string becomes the text
//! (outer loop) to minimise iterations. Score update is branchless.

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

    // Trim common prefix.  i < limit guarantees get_unchecked safety.
    let limit = a.len().min(b.len());
    let mut pre = 0usize;
    while pre < limit {
        if unsafe { *a.get_unchecked(pre) != *b.get_unchecked(pre) } { break; }
        pre += 1;
    }
    a = &a[pre..];
    b = &b[pre..];

    // Trim common suffix.  suf < limit keeps indices in bounds.
    let limit = a.len().min(b.len());
    let mut suf = 0usize;
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

    // Put the shorter string as the text (outer loop) for fewer iterations.
    // Edit distance is symmetric so swapping inputs gives the same answer.
    let (a, b) = if m < n { (b, a) } else { (a, b) };
    let m = a.len(); // pattern length (in bitmask)
    let n = b.len(); // text length  (outer loop count)

    // Build pattern bitmasks: peq[c] has bit i set iff a[i] == c.
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

    // Restore only the ≤m peq slots we dirtied.
    for &c in a.iter() {
        unsafe { *PEQ.get_unchecked_mut(c as usize) = 0; }
    }

    score
}

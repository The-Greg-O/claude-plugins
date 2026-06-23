//! Myers/Hyyrö single-word bit-parallel edit distance (Hyyrö 2001).
//! Pattern bitmask fits in one u64 since strings are guaranteed ≤ 64 bytes.
//! O(n) per call: ~11 bit-ops per text character, independent of pattern length.
//! Global peq table avoids zeroing 256×8 bytes; only the ≤m entries touched
//! are cleared after each call.

static mut BUF: [u8; 512] = [0; 512];
static mut PEQ: [u64; 256] = [0; 256];

#[no_mangle]
pub extern "C" fn input_ptr() -> u32 {
    unsafe { core::ptr::addr_of!(BUF) as u32 }
}

#[no_mangle]
pub extern "C" fn solve(a_ptr: u32, a_len: u32, b_ptr: u32, b_len: u32) -> u32 {
    let a = unsafe { core::slice::from_raw_parts(a_ptr as *const u8, a_len as usize) };
    let b = unsafe { core::slice::from_raw_parts(b_ptr as *const u8, b_len as usize) };

    let m = a.len();
    let n = b.len();
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }

    // Build pattern match bitmasks: peq[c] has bit i set iff a[i] == c.
    // Using the global static to skip zero-initialising all 256 slots.
    for (i, &c) in a.iter().enumerate() {
        unsafe { *PEQ.get_unchecked_mut(c as usize) |= 1u64 << i; }
    }

    // Hyyrö 2001 main loop.
    // Pv/Mv encode +1/-1 vertical deltas for the current text column.
    let mut pv: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut mv: u64 = 0;
    let mut score = m as u32;       // distance(a, b[0..0]) = m
    let high = 1u64 << (m - 1);    // sentinel bit: tracks the bottom row

    for &c in b.iter() {
        let eq = unsafe { *PEQ.get_unchecked(c as usize) };
        let xv = eq | mv;
        // Carry-propagation produces the horizontal delta bitvector.
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = mv | !(xh | pv);  // positive horizontal deltas
        let mh = pv & xh;           // negative horizontal deltas
        // Update score from the bottom row.
        if ph & high != 0 { score += 1; }
        if mh & high != 0 { score -= 1; }
        // Shift horizontal deltas into vertical for next column.
        let ph_s = (ph << 1) | 1;  // shift in 1: top row always +1
        let mh_s = mh << 1;
        pv = mh_s | !(xv | ph_s);
        mv = ph_s & xv;
    }

    // Restore global peq to zero (only the entries we set).
    for &c in a.iter() {
        unsafe { *PEQ.get_unchecked_mut(c as usize) = 0; }
    }

    score
}

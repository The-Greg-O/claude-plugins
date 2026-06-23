//! cpq02: cpq01 + explicit pointer-based Myers outer loop (no counter local).
//!
//! cpq01's inner loop compiles to 71 instructions, with 9 for loop management:
//!   local.get b_ptr; i32.const 1; i32.add; local.set b_ptr   (4)
//!   local.get count; i32.const -1; i32.add; local.tee count   (4)
//!   br_if 0                                                    (1)
//! Replacing the counter with an end-pointer comparison:
//!   i32.const 1; local.get b_ptr; i32.add; local.tee b_ptr   (4)
//!   local.get b_end; i32.ne                                   (2)
//!   br_if 0                                                    (1)
//! saves 2 instructions per Myers iteration (expected ~2,500 fuel).
//! All other logic is identical to cpq01.

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

    let mut bit = 1u64;
    for &c in a.iter() {
        unsafe { *peq_s.add(c as usize) |= bit; }
        bit = bit.wrapping_shl(1);
    }

    let mut pv: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut mv: u64 = 0u64;
    let mut score = m as u32;
    let high = 1u64 << (m - 1);

    // Do-while loop via raw pointer: advance+compare at BOTTOM forces LLVM
    // to emit pointer-comparison termination instead of a counter decrement.
    // Expected: local.tee bp; get b_end; ne; br_if (saves 2 instrs/iter vs counter).
    // Safe: n >= 1 checked above (n==0 returns early), so loop body runs >= 1 time.
    unsafe {
        let mut bp = b.as_ptr();
        let b_end = bp.add(b.len());
        loop {
            let c = *bp;
            let eq = *peq_s.add(c as usize);
            let xv = eq | mv;
            let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
            let ph = mv | !(xh | pv);
            let mh = pv & xh;
            score = score.wrapping_add(
                ((ph & high != 0) as i32 - (mh & high != 0) as i32) as u32,
            );
            let ph_s = (ph << 1) | 1;
            pv = (mh << 1) | !(xv | ph_s);
            mv = ph_s & xv;
            bp = bp.add(1);
            if bp == b_end { break; }
        }
    }

    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }
    score
}

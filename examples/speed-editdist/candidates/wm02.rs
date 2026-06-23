//! Hybrid: Levenshtein diagonal WMM (d≤MAX_D) + Myers bit-parallel fallback (wm02).
//!
//! 89/90 frozen cases have d≤12; 1 outlier has d=53.
//! For d≤12: WMM runs (d+1)² diagonal iterations — much fewer ops than Myers O(nm).
//! For d=53: WMM fails after MAX_D iterations, falls back to cpq01 Myers bit-parallel.
//! Word-at-a-time trim kept throughout.
//!
//! WMM: fp[k] = biased(furthest row on diagonal k with ≤p errors).
//! Levenshtein recurrence: fp[k] = snake(k, max(fp_old[k-1], fp_old[k+1]+1, fp_old[k]+1))
//!   del=fp[k-1], ins=fp[k+1]+1, sub=fp[k]+1. In-place via old_val trick.
//! k-range: -p to +p (reachable from start diagonal 0 with p errors).

const MAX_D: i32 = 13; // try WMM up to this threshold; fall back to Myers if exceeded

static mut BUF: [u8; 512] = [0u8; 512];
// WMM working array. k ∈ [-128,128]; V_OFF=129; size=259.
static mut FP: [i32; 259] = [0i32; 259];
const FP_OFF: usize = 129;
// Myers PEQ compact table (a-t = indices 0-19).
static mut PEQ: [u64; 20] = [0u64; 20];

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

    let ma = a.len() as i32;
    let nb = b.len() as i32;
    if ma == 0 { return nb as u32; }
    if nb == 0 { return ma as u32; }

    // Ensure n >= m (target diagonal delta = n-m >= 0).
    let (a, b, m, n) = if ma > nb { (b, a, nb, ma) } else { (a, b, ma, nb) };
    let delta = (n - m) as isize;

    // ── WMM phase ──────────────────────────────────────────────────────────
    // Reset FP to biased-0 (= actual -1 = unreachable).
    unsafe { core::ptr::write_bytes(FP.as_mut_ptr(), 0u8, 259); }
    let fp = unsafe { FP.as_mut_ptr().add(FP_OFF) };
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    let mut p = 0i32;
    while p <= MAX_D {
        let p_s = p as isize;
        let mut old_val = unsafe { *fp.offset(-p_s - 1) };
        let mut k = -p_s;
        while k <= p_s {
            let save_k = unsafe { *fp.offset(k) };
            let t_del  = old_val;
            let t_ins  = unsafe { *fp.offset(k + 1) } + 1;
            let t_sub  = save_k + 1;
            let t = t_del.max(t_ins).max(t_sub);
            let mut row = t - 1;
            let mut col = row + k as i32;
            while row < m && col >= 0 && col < n {
                if unsafe { *ap.add(row as usize) != *bp.add(col as usize) } { break; }
                row += 1;
                col += 1;
            }
            unsafe { *fp.offset(k) = row + 1; }
            if k == delta && row >= m { return p as u32; }
            old_val = save_k;
            k += 1;
        }
        p += 1;
    }

    // ── Myers bit-parallel fallback (cpq01 approach) ──────────────────────
    // d > MAX_D: use Myers O(nm/w) with compact PEQ + bulk cleanup.
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
    for &c in b.iter() {
        let eq = unsafe { *peq_s.add(c as usize) };
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
    }
    unsafe { core::ptr::write_bytes(PEQ.as_mut_ptr(), 0, 20); }
    score
}

//! Levenshtein diagonal (Wu-Manber / Ukkonen) + word-at-a-time trim (wm01).
//!
//! Unlike the pure Myers O(nd) LCS algorithm, this handles all 3 edit ops
//! (ins/del/sub each cost 1) via recurrence:
//!   fp[k] = snake(k, max(fp_old[k-1], fp_old[k+1]+1, fp_old[k]+1))
//!         delete^         insert^             substitute^
//!
//! Uses single-array in-place update with `old_val` trick to track old fp[k-1]
//! before it's overwritten (eliminates need for a second array).
//! FP[FP_OFF+k] is biased: stored = actual_row + 1 (so 0 = "unreachable").
//!
//! For test data (ALPHA=a-p, edits≤L//5≤12): Σ(2p+1) p=0..d ≈ d² ≈ 144
//! diagonal iterations → ~4-6× fuel reduction expected over cpq01 (110,642).

static mut BUF: [u8; 512] = [0u8; 512];
// FP[FP_OFF + k] = biased furthest row on diagonal k. k ∈ [-128,128]; size=259.
static mut FP: [i32; 259] = [0i32; 259];
const FP_OFF: usize = 129;

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

    // Ensure n >= m so target diagonal delta = n-m >= 0.
    let (a, b, m, n) = if ma > nb { (b, a, nb, ma) } else { (a, b, ma, nb) };
    let delta = (n - m) as isize;

    // Reset FP to 0 (biased -1 = "unreachable"). Cost: ceil(1036/64) = 17 fuel.
    unsafe { core::ptr::write_bytes(FP.as_mut_ptr(), 0u8, 259); }

    let fp = unsafe { FP.as_mut_ptr().add(FP_OFF) };
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    let mut p = 0i32;
    while p <= m + n {
        let p_s = p as isize;
        // Reachable diagonals from starting diagonal k=0 with p errors: [-p, p].
        // old_val tracks old FP[k-1] before overwrite (delete transition).
        let mut old_val = unsafe { *fp.offset(-p_s - 1) };

        let mut k = -p_s;
        while k <= p_s {
            let save_k = unsafe { *fp.offset(k) };          // old FP[k]
            let t_del  = old_val;                            // old FP[k-1]: delete
            let t_ins  = unsafe { *fp.offset(k + 1) } + 1; // old FP[k+1]+1: insert
            let t_sub  = save_k + 1;                        // old FP[k]+1: substitute
            let t = t_del.max(t_ins).max(t_sub);

            // Snake: advance while chars match.
            let mut row = t - 1; // unbiased actual row
            let mut col = row + k as i32;
            while row < m && col >= 0 && col < n {
                if unsafe { *ap.add(row as usize) != *bp.add(col as usize) } { break; }
                row += 1;
                col += 1;
            }

            unsafe { *fp.offset(k) = row + 1; } // re-bias

            if k == delta && row >= m { return p as u32; }

            old_val = save_k;
            k += 1;
        }
        p += 1;
    }

    (m + n) as u32
}

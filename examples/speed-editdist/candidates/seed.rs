//! Seed: textbook two-row Wagner-Fischer edit distance — the naive O(n*m) DP
//! everyone recognizes. Correct but slow. The loop's job is to beat it on fuel
//! (target: Myers 1999 bit-parallel, ~O(n) for patterns up to one machine word).
//!
//! Experiment ABI (do not change the export names/signatures):
//!   input_ptr() -> u32                          where the host writes a||b
//!   solve(a_ptr,a_len,b_ptr,b_len) -> u32        the edit distance
//! Compiles to wasm32 with NO host imports, so it can only compute.

const MAXLEN: usize = 1024;
static mut BUF: [u8; 2 * MAXLEN] = [0; 2 * MAXLEN];

#[no_mangle]
pub extern "C" fn input_ptr() -> u32 {
    unsafe { core::ptr::addr_of!(BUF) as u32 }
}

#[no_mangle]
pub extern "C" fn solve(a_ptr: u32, a_len: u32, b_ptr: u32, b_len: u32) -> u32 {
    let a = unsafe { core::slice::from_raw_parts(a_ptr as *const u8, a_len as usize) };
    let b = unsafe { core::slice::from_raw_parts(b_ptr as *const u8, b_len as usize) };
    let (n, m) = (a.len(), b.len());

    let mut prev = [0u32; MAXLEN + 1];
    let mut curr = [0u32; MAXLEN + 1];
    for j in 0..=m {
        prev[j] = j as u32;
    }
    for i in 1..=n {
        curr[0] = i as u32;
        let ai = a[i - 1];
        for j in 1..=m {
            let cost = if ai == b[j - 1] { 0 } else { 1 };
            let mut v = prev[j] + 1;
            if curr[j - 1] + 1 < v {
                v = curr[j - 1] + 1;
            }
            if prev[j - 1] + cost < v {
                v = prev[j - 1] + cost;
            }
            curr[j] = v;
        }
        core::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

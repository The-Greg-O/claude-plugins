// fuel_host — deterministic cost meter for the edit-distance speed loop.
//
// Loads a candidate wasm module (exports `memory`, `input_ptr() -> u32`, and
// `solve(a_ptr,a_len,b_ptr,b_len) -> u32`), runs it against the test vectors,
// and reports byte-exact correctness + total Wasmtime/wasmi FUEL consumed by
// `solve` (the deterministic per-instruction count). The candidate is given no
// host imports, so it can only compute — the wasm boundary is the sandbox.
//
//   fuel_host <candidate.wasm> <vectors.bin>   ->   one JSON line
use std::{env, fs};
use wasmi::{Config, Engine, Linker, Module, Store};

fn rd_u32(buf: &[u8], pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    v
}

fn main() {
    let wasm_path = env::args().nth(1).expect("usage: fuel_host <wasm> <vectors.bin>");
    let vec_path = env::args().nth(2).expect("usage: fuel_host <wasm> <vectors.bin>");
    let wasm = fs::read(&wasm_path).unwrap();
    let vbuf = fs::read(&vec_path).unwrap();

    let mut pos = 0usize;
    let count = rd_u32(&vbuf, &mut pos);
    let mut cases: Vec<(Vec<u8>, Vec<u8>, u32)> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let alen = rd_u32(&vbuf, &mut pos) as usize;
        let a = vbuf[pos..pos + alen].to_vec();
        pos += alen;
        let blen = rd_u32(&vbuf, &mut pos) as usize;
        let b = vbuf[pos..pos + blen].to_vec();
        pos += blen;
        let exp = rd_u32(&vbuf, &mut pos);
        cases.push((a, b, exp));
    }

    let mut config = Config::default();
    config.consume_fuel(true);
    let engine = Engine::new(&config);
    let module = match Module::new(&engine, &wasm[..]) {
        Ok(m) => m,
        Err(e) => return println!("{{\"correct\":false,\"error\":\"module: {e}\"}}"),
    };
    let mut store = Store::new(&engine, ());
    let linker: Linker<()> = Linker::new(&engine);
    let instance = match linker.instantiate(&mut store, &module).and_then(|p| p.start(&mut store)) {
        Ok(i) => i,
        Err(e) => return println!("{{\"correct\":false,\"error\":\"instantiate: {e}\"}}"),
    };
    let memory = instance.get_memory(&store, "memory").expect("module must export memory");
    let input_ptr = instance
        .get_typed_func::<(), u32>(&store, "input_ptr")
        .expect("module must export input_ptr() -> u32");
    let solve = instance
        .get_typed_func::<(u32, u32, u32, u32), u32>(&store, "solve")
        .expect("module must export solve(u32,u32,u32,u32) -> u32");

    let big: u64 = 100_000_000_000;
    store.set_fuel(big).unwrap();
    let base = input_ptr.call(&mut store, ()).unwrap();
    let mut total_fuel: u64 = 0;
    for (i, (a, b, exp)) in cases.iter().enumerate() {
        memory.write(&mut store, base as usize, a).unwrap();
        memory.write(&mut store, base as usize + a.len(), b).unwrap();
        store.set_fuel(big).unwrap();
        let r = match solve.call(
            &mut store,
            (base, a.len() as u32, base + a.len() as u32, b.len() as u32),
        ) {
            Ok(v) => v,
            Err(_) => return println!(
                "{{\"correct\":false,\"total_fuel\":{total_fuel},\"n\":{},\"first_bad\":{i},\"reason\":\"trap\"}}",
                cases.len()
            ),
        };
        total_fuel += big - store.get_fuel().unwrap();
        if r != *exp {
            return println!(
                "{{\"correct\":false,\"total_fuel\":{total_fuel},\"n\":{},\"first_bad\":{i},\"got\":{r},\"want\":{exp}}}",
                cases.len()
            );
        }
    }
    println!("{{\"correct\":true,\"total_fuel\":{total_fuel},\"n\":{}}}", cases.len());
}

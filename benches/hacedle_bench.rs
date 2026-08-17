use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hacedle::{dot_i8, dot_i8_simd, dot_batch, mmap_skb, SKB_TOTAL_BYTES, TensorI8};
use std::fs;
use std::path::PathBuf;

fn build_tensor(len: usize, seed: i8) -> TensorI8 {
    let mut data = vec![0i8; len];
    let mut i = 0usize;
    while i < len {
        data[i] = seed.wrapping_add(i as i8);
        i += 1;
    }
    TensorI8::new(data)
}

fn temp_skb_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    path.push(format!("hacedle_bench_{}.skb", pid));
    path
}

fn ensure_skb_file(path: &PathBuf) {
    if path.exists() {
        return;
    }
    let mut bytes = vec![0u8; SKB_TOTAL_BYTES];
    let mut i = 0usize;
    while i < bytes.len() {
        bytes[i] = (i % 251) as u8;
        i += 1;
    }
    let _ = fs::write(path, bytes);
}

fn bench_dot(c: &mut Criterion) {
    let a = build_tensor(1024, 1);
    let b = build_tensor(1024, 2);

    c.bench_function("dot_scalar", |bencher| {
        bencher.iter(|| black_box(dot_i8(&a, &b)))
    });

    c.bench_function("dot_simd", |bencher| {
        bencher.iter(|| black_box(dot_i8_simd(&a, &b)))
    });
}

fn bench_batch_dot(c: &mut Criterion) {
    let a = build_tensor(2048, 3);
    let b = build_tensor(2048, 4);
    c.bench_function("batch_dot", |bencher| {
        bencher.iter(|| black_box(dot_batch(&a.data, &b.data)))
    });
}

fn bench_mmap(c: &mut Criterion) {
    let path = temp_skb_path();
    ensure_skb_file(&path);

    c.bench_function("skb_mmap", |bencher| {
        bencher.iter(|| {
            let mapped = mmap_skb(&path).unwrap();
            black_box(mapped);
        })
    });
}

criterion_group!(benches, bench_dot, bench_batch_dot, bench_mmap);
criterion_main!(benches);

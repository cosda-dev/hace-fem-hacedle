use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

const MAX_STR_LEN: u64 = 1_000_000;
const MAX_N_DIMS: u32 = 16;
const MAX_DIM: u64 = 1_000_000;
const MAX_ALLOC: u64 = 8 * 1024 * 1024 * 1024;
const GGUF_DEFAULT_ALIGNMENT: u32 = 32;

struct AuditReader {
    inner: BufReader<File>,
    pos: u64,
}

impl AuditReader {
    fn new(file: File) -> Self {
        Self {
            inner: BufReader::new(file),
            pos: 0,
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.inner.read_exact(buf)?;
        self.pos = self.pos.saturating_add(buf.len() as u64);
        Ok(())
    }

    fn read_u32_le(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64_le(&mut self) -> std::io::Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        if n > (i64::MAX as u64) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Cannot skip {n} bytes (too large)"),
            ));
        }
        self.inner.seek(SeekFrom::Current(n as i64))?;
        self.pos = self.pos.saturating_add(n);
        Ok(())
    }

    fn read_gguf_string(&mut self) -> std::io::Result<String> {
        let len = self.read_u64_le()?;
        if len > MAX_STR_LEN {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("GGUF string length too large: {len}"),
            ));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
}

#[derive(Debug, Clone)]
struct TensorHeader {
    hdr_offset: u64,
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    offset_in_data: u64,
}

fn pad_to(align: u32, pos: u64) -> u64 {
    if align == 0 {
        return pos;
    }
    let a = align as u64;
    let rem = pos % a;
    if rem == 0 { pos } else { pos + (a - rem) }
}

fn ggml_type_name(ty: u32) -> &'static str {
    match ty {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_K",
        11 => "q3_K",
        12 => "q4_K",
        13 => "q5_K",
        14 => "q6_K",
        15 => "q8_K",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        30 => "bf16",
        34 => "tq1_0",
        35 => "tq2_0",
        39 => "mxfp4",
        40 => "nvfp4",
        41 => "q1_0",
        _ => "unknown",
    }
}

fn ggml_type_traits(ty: u32) -> Option<(u32, u32)> {
    Some(match ty {
        0 => (1, 4),
        1 => (1, 2),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        30 => (1, 2),
        12 => (256, 184), // block_q4_K = 184 bytes (d+dmin+scales+qs+qh)
        13 => (256, 176),
        14 => (256, 210),
        _ => return None,
    })
}

fn ggml_nbytes(ty: u32, dims: &[u64]) -> Result<u64, Box<dyn std::error::Error>> {
    let (blck, type_size) = ggml_type_traits(ty).ok_or_else(|| {
        format!("ggml_type unsupported for byte_size: {ty} ({})", ggml_type_name(ty))
    })?;

    let blck = blck as u64;
    let type_size = type_size as u64;

    let ne0 = dims.first().copied().unwrap_or(0);
    if ne0 == 0 {
        return Err("ggml_nbytes: ne0 is 0".into());
    }
    if ne0 % blck != 0 {
        return Err(format!(
            "ggml_nbytes: ne0 ({ne0}) must be multiple of blck_size ({blck}) for type {}",
            ggml_type_name(ty)
        )
        .into());
    }

    let rows: u128 = dims.iter().skip(1).fold(1u128, |acc, &v| acc.saturating_mul(v as u128));
    let row_bytes: u128 = (type_size as u128).saturating_mul((ne0 / blck) as u128);
    let total: u128 = row_bytes.saturating_mul(rows);
    if total > u64::MAX as u128 {
        return Err("ggml_nbytes: overflow".into());
    }
    Ok(total as u64)
}

const K_SCALE_SIZE: usize = 12;

#[repr(C, align(1))]
#[derive(Clone, Copy)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; K_SCALE_SIZE],
    qs: [u8; 256 / 2],
} // sizeof = 2 + 2 + 12 + 128 = 144 bytes

#[repr(C, align(1))]
#[derive(Clone, Copy)]
struct BlockQ6K {
    ql: [u8; 256 / 2],
    qh: [u8; 256 / 4],
    scales: [i8; 256 / 16],
    d: u16,
}

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x03FF) as u32;

    let f_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e: i32 = -14;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let exp_f32 = ((e + 127) as u32) << 23;
            let mant_f32 = m << 13;
            sign | exp_f32 | mant_f32
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let exp_f32 = ((exp as i32 - 15 + 127) as u32) << 23;
        let mant_f32 = mant << 13;
        sign | exp_f32 | mant_f32
    };

    f32::from_bits(f_bits)
}

fn dequantize_row_q4_k(row: &[u8], k: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    const QK_K: usize = 256;
    const BLK_SIZE_Q4K: usize = 144; // d(2)+dmin(2)+scales(12)+qs(128) = 144 bytes

    if k % QK_K != 0 {
        return Err(format!("q4_K requires k multiple of {QK_K}, got {k}").into());
    }
    let nb = k / QK_K;
    let expected_bytes = nb * BLK_SIZE_Q4K;
    if row.len() < expected_bytes {
        return Err(format!(
            "q4_K row byte size mismatch: got {} expected {}",
            row.len(),
            expected_bytes
        )
        .into());
    }

    let mut y = vec![0.0f32; k];

    for i in 0..nb {
        let blk_bytes = &row[i * BLK_SIZE_Q4K..(i + 1) * BLK_SIZE_Q4K];
        let blk: BlockQ4K = unsafe { std::ptr::read_unaligned(blk_bytes.as_ptr() as *const BlockQ4K) };

        let d = fp16_to_f32(blk.d);
        let min = fp16_to_f32(blk.dmin);
        let mut q_off = 0usize;
        let mut is = 0usize;

        for j in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is + 0, &blk.scales);
            let d1 = d * (sc as f32);
            let m1 = min * (m as f32);
            
            let (sc, m) = get_scale_min_k4(is + 1, &blk.scales);
            let d2 = d * (sc as f32);
            let m2 = min * (m as f32);

            for l in 0..32usize {
                y[j + l] = d1 * ((blk.qs[q_off + l] & 0xF) as f32) - m1;
                y[j + l + 32] = d2 * ((blk.qs[q_off + l] >> 4) as f32) - m2;
            }

            q_off += 32;
            is += 2;
        }
    }

    Ok(y)
}

fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) & 0x3F;
        let m = ((scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4)) & 0x3F;
        (sc, m)
    }
}

fn dequantize_row_q6_k(row: &[u8], k: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    const QK_K: usize = 256;
    const BLK_SIZE: usize = 210;

    if k % QK_K != 0 {
        return Err(format!("q6_K requires k multiple of {QK_K}, got {k}").into());
    }
    let nb = k / QK_K;
    if row.len() != nb * BLK_SIZE {
        return Err(format!(
            "q6_K row byte size mismatch: got {} expected {}",
            row.len(),
            nb * BLK_SIZE
        )
        .into());
    }

    let mut y = vec![0.0f32; k];
    let mut y_off = 0usize;

    for i in 0..nb {
        let blk_bytes = &row[i * BLK_SIZE..(i + 1) * BLK_SIZE];
        let blk: BlockQ6K = unsafe { std::ptr::read_unaligned(blk_bytes.as_ptr() as *const BlockQ6K) };

        let d = fp16_to_f32(blk.d);
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for _ in (0..QK_K).step_by(128) {
            for l in 0..32usize {
                let is = l / 16;
                let qh = blk.qh[qh_off + l];
                let ql0 = blk.ql[ql_off + l + 0];
                let ql32 = blk.ql[ql_off + l + 32];

                let q1 = (((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((ql32 & 0xF) | (((qh >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((ql0 >> 4) | (((qh >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((ql32 >> 4) | (((qh >> 6) & 3) << 4)) as i8) - 32;

                y[y_off + l + 0] = d * (blk.scales[sc_off + is + 0] as f32) * (q1 as f32);
                y[y_off + l + 32] = d * (blk.scales[sc_off + is + 2] as f32) * (q2 as f32);
                y[y_off + l + 64] = d * (blk.scales[sc_off + is + 4] as f32) * (q3 as f32);
                y[y_off + l + 96] = d * (blk.scales[sc_off + is + 6] as f32) * (q4 as f32);
            }

            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }

    Ok(y)
}

#[cfg(windows)]
fn mmap_read_first_bytes(path: &str, file_offset: u64, len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    type HANDLE = *mut c_void;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x0004;

    #[repr(C)]
    struct SYSTEM_INFO {
        wProcessorArchitecture: u16,
        wReserved: u16,
        dwPageSize: u32,
        lpMinimumApplicationAddress: *mut c_void,
        lpMaximumApplicationAddress: *mut c_void,
        dwActiveProcessorMask: usize,
        dwNumberOfProcessors: u32,
        dwProcessorType: u32,
        dwAllocationGranularity: u32,
        wProcessorLevel: u16,
        wProcessorRevision: u16,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
        fn CreateFileMappingW(
            hFile: HANDLE,
            lpFileMappingAttributes: *mut c_void,
            flProtect: u32,
            dwMaximumSizeHigh: u32,
            dwMaximumSizeLow: u32,
            lpName: *const u16,
        ) -> HANDLE;
        fn MapViewOfFile(
            hFileMappingObject: HANDLE,
            dwDesiredAccess: u32,
            dwFileOffsetHigh: u32,
            dwFileOffsetLow: u32,
            dwNumberOfBytesToMap: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> i32;
        fn CloseHandle(hObject: HANDLE) -> i32;
    }

    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    if file_offset > file_size {
        return Err(format!("mmap_read: file_offset {file_offset} > file_size {file_size}").into());
    }

    let mut si = SYSTEM_INFO {
        wProcessorArchitecture: 0,
        wReserved: 0,
        dwPageSize: 0,
        lpMinimumApplicationAddress: std::ptr::null_mut(),
        lpMaximumApplicationAddress: std::ptr::null_mut(),
        dwActiveProcessorMask: 0,
        dwNumberOfProcessors: 0,
        dwProcessorType: 0,
        dwAllocationGranularity: 0,
        wProcessorLevel: 0,
        wProcessorRevision: 0,
    };
    unsafe { GetSystemInfo(&mut si as *mut SYSTEM_INFO) };
    let gran = si.dwAllocationGranularity as u64;
    if gran == 0 {
        return Err("mmap_read: allocation granularity is 0".into());
    }

    let map_offset = file_offset / gran * gran;
    let delta = (file_offset - map_offset) as usize;
    let want = len.min((file_size - file_offset) as usize);
    let map_len = delta.saturating_add(want);

    let hfile = file.as_raw_handle() as HANDLE;
    let hmap = unsafe { CreateFileMappingW(hfile, std::ptr::null_mut(), PAGE_READONLY, 0, 0, std::ptr::null()) };
    if hmap.is_null() {
        return Err("CreateFileMappingW failed".into());
    }

    let view = unsafe {
        MapViewOfFile(
            hmap,
            FILE_MAP_READ,
            (map_offset >> 32) as u32,
            (map_offset & 0xFFFF_FFFF) as u32,
            map_len,
        )
    };

    if view.is_null() {
        unsafe { CloseHandle(hmap) };
        return Err("MapViewOfFile failed".into());
    }

    let mut out = vec![0u8; want];
    unsafe {
        let src = (view as *const u8).add(delta);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), want);
        UnmapViewOfFile(view as *const c_void);
        CloseHandle(hmap);
    }

    Ok(out)
}

#[cfg(not(windows))]
fn mmap_read_first_bytes(_path: &str, _file_offset: u64, _len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("mmap_read_first_bytes is only supported on Windows in this tool".into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Uint8 => "uint8",
            Self::Int8 => "int8",
            Self::Uint16 => "uint16",
            Self::Int16 => "int16",
            Self::Uint32 => "uint32",
            Self::Int32 => "int32",
            Self::Float32 => "float32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::Uint64 => "uint64",
            Self::Int64 => "int64",
            Self::Float64 => "float64",
        }
    }
}

fn skip_value(reader: &mut AuditReader, ty: GgufValueType) -> Result<(), Box<dyn std::error::Error>> {
    match ty {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
            reader.skip(1)?;
        }
        GgufValueType::Uint16 | GgufValueType::Int16 => {
            reader.skip(2)?;
        }
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
            reader.skip(4)?;
        }
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
            reader.skip(8)?;
        }
        GgufValueType::String => {
            let len = reader.read_u64_le()?;
            if len > MAX_STR_LEN {
                return Err(format!("GGUF string length too large: {len}").into());
            }
            reader.skip(len)?;
        }
        GgufValueType::Array => {
            let elem_ty_raw = reader.read_u32_le()?;
            let elem_ty = GgufValueType::from_u32(elem_ty_raw).ok_or_else(|| {
                format!("GGUF array element type invalid: {elem_ty_raw}")
            })?;
            let n = reader.read_u64_le()?;
            if n > MAX_STR_LEN {
                return Err(format!("GGUF array length too large: {n}").into());
            }
            match elem_ty {
                GgufValueType::String => {
                    for _ in 0..n {
                        let slen = reader.read_u64_le()?;
                        if slen > MAX_STR_LEN {
                            return Err(format!("GGUF string length too large: {slen}").into());
                        }
                        reader.skip(slen)?;
                    }
                }
                GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
                    reader.skip(n)?;
                }
                GgufValueType::Uint16 | GgufValueType::Int16 => {
                    reader.skip(n.saturating_mul(2))?;
                }
                GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
                    reader.skip(n.saturating_mul(4))?;
                }
                GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
                    reader.skip(n.saturating_mul(8))?;
                }
                GgufValueType::Array => {
                    return Err("Nested GGUF arrays are not supported by this audit tool".into());
                }
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let default_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    let path = std::env::args().nth(1).unwrap_or_else(|| default_path.to_string());

    let kv_report_path = std::path::PathBuf::from("kv_offset_report.txt");
    let tensor_report_path = std::path::PathBuf::from("tensor_header_report.txt");
    let first10_path = std::path::PathBuf::from("first_10_tensor_headers.txt");

    let file_size = std::fs::metadata(&path)?.len();

    let mut reader = AuditReader::new(File::open(&path)?);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(format!("Invalid GGUF magic: {:?}", magic).into());
    }

    let version = reader.read_u32_le()?;
    let tensor_count = reader.read_u64_le()?;
    let kv_count = reader.read_u64_le()?;

    println!("=== GGUF-PARSER-RECOVERY ===");
    println!("path={}", path);
    println!("magic=GGUF");
    println!("version={}", version);
    println!("tensor_count={}", tensor_count);
    println!("kv_count={}", kv_count);

    if version > 16 {
        println!(
            "warning: version={} looks unusual; if other fields also look wrong, suspect endian/offset drift",
            version
        );
    }

    let mut kv_report = BufWriter::new(File::create(&kv_report_path)?);
    writeln!(kv_report, "GGUF KV Offset Report")?;
    writeln!(kv_report, "path={}", path)?;
    writeln!(kv_report, "version={}", version)?;
    writeln!(kv_report, "kv_count={}", kv_count)?;
    writeln!(kv_report)?;

    let mut gguf_alignment: u32 = GGUF_DEFAULT_ALIGNMENT;

    let expected_kv = kv_count as usize;
    for i in 0..expected_kv {
        let entry_offset = reader.pos;

        let key = reader.read_gguf_string()?;
        let ty_raw = reader.read_u32_le()?;
        let ty = GgufValueType::from_u32(ty_raw).ok_or_else(|| {
            format!(
                "GGUF value type invalid at KV[{i}] offset={entry_offset}: {ty_raw}"
            )
        })?;

        if key == "general.alignment" && ty == GgufValueType::Uint32 {
            let v = reader.read_u32_le()?;
            if v == 0 || (v & (v - 1)) != 0 {
                return Err(format!("Invalid general.alignment={v} (must be power of two)").into());
            }
            gguf_alignment = v;
        } else {
            skip_value(&mut reader, ty)?;
        }
        let next_offset = reader.pos;
        writeln!(
            kv_report,
            "KV[{i}] offset={entry_offset} key={key} type={} next={next_offset}",
            ty.as_str()
        )?;
    }

    let kv_end = reader.pos;
    println!("parsed_kv={}/{} kv_end={}", expected_kv, expected_kv, kv_end);

    let tensor_directory_start = reader.pos;
    println!("tensor_directory_start={}", tensor_directory_start);

    let mut tensor_report = BufWriter::new(File::create(&tensor_report_path)?);
    let mut first10 = BufWriter::new(File::create(&first10_path)?);

    writeln!(tensor_report, "GGUF Tensor Header Report")?;
    writeln!(tensor_report, "path={}", path)?;
    writeln!(tensor_report, "tensor_count={}", tensor_count)?;
    writeln!(tensor_report, "tensor_directory_start={}", tensor_directory_start)?;
    writeln!(tensor_report)?;

    writeln!(first10, "First 10 GGUF tensor headers")?;
    writeln!(first10, "path={}", path)?;
    writeln!(first10, "tensor_directory_start={}", tensor_directory_start)?;
    writeln!(first10)?;

    let mut token_embd: Option<TensorHeader> = None;

    let expected_tensors = tensor_count as usize;
    for t in 0..expected_tensors {
        let hdr_offset = reader.pos;

        let name = reader.read_gguf_string()?;
        let n_dims = reader.read_u32_le()?;
        if n_dims > MAX_N_DIMS {
            return Err(format!(
                "GGUF dim sanity failed: tensor[{t}] hdr_offset={hdr_offset} name={name} n_dims={n_dims} (MAX_N_DIMS={MAX_N_DIMS})"
            )
            .into());
        }

        let mut dims: Vec<u64> = Vec::with_capacity(n_dims as usize);
        for d in 0..(n_dims as usize) {
            let dim = reader.read_u64_le()?;
            if dim > MAX_DIM {
                return Err(format!(
                    "GGUF dim sanity failed: tensor[{t}] hdr_offset={hdr_offset} name={name} dims[{d}]={dim} (MAX_DIM={MAX_DIM})"
                )
                .into());
            }
            dims.push(dim);
        }

        let ggml_type = reader.read_u32_le()?;
        let offset = reader.read_u64_le()?;

        if token_embd.is_none() && name == "token_embd.weight" {
            token_embd = Some(TensorHeader {
                hdr_offset,
                name: name.clone(),
                dims: dims.clone(),
                ggml_type,
                offset_in_data: offset,
            });
        }

        writeln!(
            tensor_report,
            "Tensor[{t}] hdr_offset={hdr_offset} name={name} n_dims={n_dims} dims={:?} ggml_type={ggml_type} offset={offset}",
            dims
        )?;

        if t < 10 {
            writeln!(
                first10,
                "Tensor[{t}] hdr_offset={hdr_offset} name={name} n_dims={n_dims} dims={:?} ggml_type={ggml_type} offset={offset}",
                dims
            )?;
        }

        let mut dim_prod: u128 = 1;
        for &dim in &dims {
            dim_prod = dim_prod.saturating_mul(dim as u128);
        }
        let bytes_assuming_f32 = dim_prod.saturating_mul(4) as u64;
        if bytes_assuming_f32 > MAX_ALLOC {
            return Err(format!(
                "GGUF allocation sanity failed (audit-only guard): tensor[{t}] name={name} dims={:?} bytes_assuming_f32={bytes_assuming_f32} (MAX_ALLOC={MAX_ALLOC})",
                dims
            )
            .into());
        }
    }

    let tensor_dir_end = reader.pos;
    println!(
        "parsed_tensor_headers={}/{} tensor_dir_end={}",
        expected_tensors, expected_tensors, tensor_dir_end
    );

    let data_section_start = if tensor_count > 0 {
        pad_to(gguf_alignment, tensor_dir_end)
    } else {
        tensor_dir_end
    };

    if let Some(t0) = token_embd {
        let token_bytes = ggml_nbytes(t0.ggml_type, &t0.dims)?;
        let token_file_offset = data_section_start.saturating_add(t0.offset_in_data);
        if token_file_offset > file_size {
            return Err(format!(
                "token_embd.weight file_offset {token_file_offset} > file_size {file_size}"
            )
            .into());
        }
        if token_file_offset.saturating_add(token_bytes) > file_size {
            return Err(format!(
                "token_embd.weight range [{token_file_offset}..{}] exceeds file_size {file_size}",
                token_file_offset.saturating_add(token_bytes)
            )
            .into());
        }

        let first64 = mmap_read_first_bytes(&path, token_file_offset, 64)?;
        let first64_hex = first64
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");

        writeln!(first10)?;
        writeln!(first10, "P1 token_embd.weight projection")?;
        writeln!(first10, "gguf_alignment={}", gguf_alignment)?;
        writeln!(first10, "tensor_dir_end={}", tensor_dir_end)?;
        writeln!(first10, "data_section_start={}", data_section_start)?;
        writeln!(first10, "tensor_hdr_offset={}", t0.hdr_offset)?;
        writeln!(first10, "tensor_name={}", t0.name)?;
        writeln!(first10, "dims={:?}", t0.dims)?;
        writeln!(first10, "ggml_type={} ({})", t0.ggml_type, ggml_type_name(t0.ggml_type))?;
        writeln!(first10, "offset_in_data={}", t0.offset_in_data)?;
        writeln!(first10, "file_offset={}", token_file_offset)?;
        writeln!(first10, "byte_size={}", token_bytes)?;
        writeln!(first10, "first_64_bytes_hex={}", first64_hex)?;

        println!("P1_token_embd_weight:");
        println!("  gguf_alignment={}", gguf_alignment);
        println!("  tensor_dir_end={}", tensor_dir_end);
        println!("  data_section_start={}", data_section_start);
        println!("  ggml_type={} ({})", t0.ggml_type, ggml_type_name(t0.ggml_type));
        println!("  offset_in_data={}", t0.offset_in_data);
        println!("  file_offset={}", token_file_offset);
        println!("  byte_size={}", token_bytes);
        println!("  first_64_bytes_hex={}", first64_hex);

        let token_id: u64 = 42;
        if t0.dims.len() != 2 {
            return Err(format!("P2 expects token_embd.weight dims=2, got {:?}", t0.dims).into());
        }
        let emb_len = t0.dims[0];
        let vocab = t0.dims[1];
        if token_id >= vocab {
            return Err(format!("token_id {token_id} out of range vocab={vocab}").into());
        }

        let (blck, type_size) = ggml_type_traits(t0.ggml_type).ok_or_else(|| {
            format!(
                "P2 unsupported ggml_type {} ({})",
                t0.ggml_type,
                ggml_type_name(t0.ggml_type)
            )
        })?;
        let blck = blck as u64;
        let type_size = type_size as u64;
        if emb_len % blck != 0 {
            return Err(format!("embedding_length {emb_len} not multiple of blck_size {blck}").into());
        }
        let row_bytes = type_size.saturating_mul(emb_len / blck);
        let row_file_offset = token_file_offset.saturating_add(row_bytes.saturating_mul(token_id));

        let row_data = mmap_read_first_bytes(&path, row_file_offset, row_bytes as usize)?;

        let embedding = match t0.ggml_type {
            12 => dequantize_row_q4_k(&row_data, emb_len as usize)?,
            14 => dequantize_row_q6_k(&row_data, emb_len as usize)?,
            _ => {
                return Err(format!(
                    "P2 unsupported ggml_type {} ({})",
                    t0.ggml_type,
                    ggml_type_name(t0.ggml_type)
                )
                .into());
            }
        };
        let finite = embedding.iter().all(|v| v.is_finite());

        writeln!(first10)?;
        writeln!(first10, "P2 real embedding extraction (audit)")?;
        writeln!(first10, "token_id={}", token_id)?;
        writeln!(first10, "embedding_len={}", embedding.len())?;
        writeln!(first10, "finite={}", finite)?;
        writeln!(first10, "row_file_offset={}", row_file_offset)?;
        writeln!(first10, "row_byte_size={}", row_bytes)?;
        writeln!(
            first10,
            "embedding_first_8={:?}",
            embedding.iter().take(8).collect::<Vec<_>>()
        )?;

        println!("P2_real_embedding:");
        println!("  token_id={}", token_id);
        println!("  embedding_len={}", embedding.len());
        println!("  finite={}", finite);
        println!("  row_file_offset={}", row_file_offset);
        println!("  row_byte_size={}", row_bytes);
    } else {
        return Err("token_embd.weight not found in tensor directory".into());
    }

    kv_report.flush()?;
    tensor_report.flush()?;
    first10.flush()?;

    println!("reports_written:");
    println!("  {}", kv_report_path.display());
    println!("  {}", tensor_report_path.display());
    println!("  {}", first10_path.display());

    Ok(())
}

use crate::Feature;

macro_rules! f {
    ($(#[doc = $doc_addition: literal])*
        struct $module: ident:: $struct_name: ident($display_name: literal): $feature_name: literal + [$($implicitly_enabled: literal),*]
        fn $example_function_name: ident
    ) => {
        Feature {
            struct_name: stringify!($struct_name),
            feature_name: $feature_name,
            directly_implicitly_enabled: &[$($implicitly_enabled),*],
            extra_docs: concat!($($doc_addition, "\n",)*),
            example_function_name: stringify!($example_function_name),
            feature_docs_name: $display_name,
            module: stringify!($module)
        }
    }
}

pub(crate) const X86_TEMPLATE: &str = include_str!("../../templates/x86.rs");

// Data taken from: https://doc.rust-lang.org/reference/attributes/codegen.html#r-attributes.codegen.target_feature.x86
// (specifically, at https://github.com/rust-lang/reference/blob/1d930e1d5a27e114b4d22a50b0b6cd3771b92e31/src/attributes/codegen.md#x86-or-x86_64)
// TODO: Do we need to add their license attribution to our license?
// TODO: Check set against https://doc.rust-lang.org/stable/std/macro.is_x86_feature_detected.html
// In particular, we're missing lahfsahf
pub(crate) const X86_FEATURES: &[Feature] = &[
    f!(
        /// [ADX] --- Multi-Precision Add-Carry Instruction Extensions
        ///
        /// [ADX]: https://en.wikipedia.org/wiki/Intel_ADX
        struct adx::Adx("ADX"): "adx" + []
        fn uses_adx
    ),
    f!(
        /// [AES] --- Advanced Encryption Standard
        ///
        /// [AES]: https://en.wikipedia.org/wiki/AES_instruction_set
        struct crypto::Aes("AES"): "aes" + ["sse2"]
        fn uses_aes
    ),
    f!(
        /// [AVX] --- Advanced Vector Extensions
        ///
        /// [AVX]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
        struct avx::Avx("AVX"): "avx" + ["sse4.2"]
        fn uses_avx
    ),
    f!(
        /// [AVX2] --- Advanced Vector Extensions 2
        ///
        /// [AVX2]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX2
        struct avx::Avx2("AVX2"): "avx2" + ["avx"]
        fn uses_avx2
    ),
    f!(
        /// [AVX512-BF16] --- Advanced Vector Extensions 512-bit - Bfloat16 Extensions
        ///
        /// [AVX512-BF16]: https://en.wikipedia.org/wiki/AVX-512#BF16
        struct avx512::Avx512bf16("AVX512-BF16"): "avx512bf16" + ["avx512bw"]
        fn uses_avx512bf16
    ),
    f!(
        /// [AVX512-BITALG] --- Advanced Vector Extensions 512-bit - Bit Algorithms
        ///
        ///
        /// [AVX512-BITALG]: https://en.wikipedia.org/wiki/AVX-512#VPOPCNTDQ_and_BITALG
        struct avx512::Avx512bitalg("AVX512-BITALG"): "avx512bitalg" + ["avx512bw"]
        fn uses_avx512bitalg
    ),
    f!(
        /// [AVX512-BW] --- Advanced Vector Extensions 512-bit - Byte and Word Instructions
        ///
        /// [AVX512-BW]: https://en.wikipedia.org/wiki/AVX-512#BW,_DQ_and_VBMI
        struct avx512::Avx512bw("AVX512-BW"): "avx512bw" + ["avx512f"]
        fn uses_avx512bw
    ),
    f!(
        /// [AVX512-CD] --- Advanced Vector Extensions 512-bit - Conflict Detection Instructions
        ///
        /// [AVX512-CD]: https://en.wikipedia.org/wiki/AVX-512#Conflict_detection
        struct avx512::Avx512cd("AVX512-CD"): "avx512cd" + ["avx512f"]
        fn uses_avx512cd
    ),
    f!(
        /// [AVX512-DQ] --- Advanced Vector Extensions 512-bit - Doubleword and Quadword Instructions
        ///
        /// [AVX512-DQ]: https://en.wikipedia.org/wiki/AVX-512#BW,_DQ_and_VBMI
        struct avx512::Avx512dq("AVX512-DQ"): "avx512dq" + ["avx512f"]
        fn uses_avx512dq
    ),
    f!(
        /// [AVX512-F] --- Advanced Vector Extensions 512-bit - Foundation
        ///
        /// [AVX512-F]: https://en.wikipedia.org/wiki/AVX-512
        struct avx512::Avx512f("AVX512-F"): "avx512f" + ["avx2", "fma", "f16c"]
        fn uses_avx512f
    ),
    f!(
        /// [AVX512-FP16] --- Advanced Vector Extensions 512-bit - Float16 Extensions
        ///
        /// [AVX512-FP16]: https://en.wikipedia.org/wiki/AVX-512#FP16
        struct avx512::Avx512fp16("AVX512-FP16"): "avx512fp16" + ["avx512bw"]
        fn uses_avx512fp16
    ),
    f!(
        /// [AVX512-IFMA] --- Advanced Vector Extensions 512-bit - Integer Fused Multiply Add
        ///
        /// [AVX512-IFMA]: https://en.wikipedia.org/wiki/AVX-512#IFMA
        struct avx512::Avx512ifma("AVX512-IFMA"): "avx512ifma" + ["avx512f"]
        fn uses_avx512ifma
    ),
    f!(
        /// [AVX512-VBMI] --- Advanced Vector Extensions 512-bit - Vector Byte Manipulation Instructions
        ///
        /// [AVX512-VBMI]: https://en.wikipedia.org/wiki/AVX-512#BW,_DQ_and_VBMI
        struct avx512::Avx512vbmi("AVX512-VBMI"): "avx512vbmi" + ["avx512bw"]
        fn uses_avx512vbmi
    ),
    f!(
        /// [AVX512-VBMI2] --- Advanced Vector Extensions 512-bit - Vector Byte Manipulation Instructions 2
        ///
        /// [AVX512-VBMI2]: https://en.wikipedia.org/wiki/AVX-512#VBMI2
        struct avx512::Avx512vbmi2("AVX512-VBMI2"): "avx512vbmi2" + ["avx512bw"]
        fn uses_avx512vbmi2
    ),
    f!(
        /// [AVX512-VL] --- Advanced Vector Extensions 512-bit - Vector Length Extensions
        ///
        /// [AVX512-VL]: https://en.wikipedia.org/wiki/AVX-512
        struct avx512::Avx512vl("AVX512-VL"): "avx512vl" + ["avx512f"]
        fn uses_avx512vl
    ),
    f!(
        /// [AVX512-VNNI] --- Advanced Vector Extensions 512-bit - Vector Neural Network Instructions
        ///
        /// [AVX512-VNNI]: https://en.wikipedia.org/wiki/AVX-512#VNNI
        struct avx512::Avx512vnni("AVX512-VNNI"): "avx512vnni" + ["avx512f"]
        fn uses_avx512vnni
    ),
    f!(
        /// [AVX512-VP2INTERSECT] --- Advanced Vector Extensions 512-bit - Vector Pair Intersection to a Pair of Mask Registers
        ///
        /// [AVX512-VP2INTERSECT]: https://en.wikipedia.org/wiki/AVX-512#VP2INTERSECT
        struct avx512::Avx512vp2intersect("AVX512-VP2INTERSECT"): "avx512vp2intersect" + ["avx512f"]
        fn uses_avx512vp2intersect
    ),
    f!(
        /// [AVX512-VPOPCNTDQ] --- Advanced Vector Extensions 512-bit - Vector Population Count Instruction
        ///
        /// [AVX512-VPOPCNTDQ]:https://en.wikipedia.org/wiki/AVX-512#VPOPCNTDQ_and_BITALG
        struct avx512::Avx512vpopcntdq("AVX512-VPOPCNTDQ"): "avx512vpopcntdq" + ["avx512f"]
        fn uses_avx512vpopcntdq
    ),
    f!(
        /// [AVX-IFMA] --- Advanced Vector Extensions - Integer Fused Multiply Add
        ///
        /// [AVX-IFMA]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA
        struct avx::Avxifma("AVX-IFMA"): "avxifma" + ["avx2"]
        fn uses_avxifma
    ),
    f!(
        /// [AVX-NE-CONVERT] --- Advanced Vector Extensions - No-Exception Floating-Point conversion Instructions
        ///
        /// [AVX-NE-CONVERT]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA
        struct avx::Avxneconvert("AVX-NE-CONVERT"): "avxneconvert" + ["avx2"]
        fn uses_avxneconvert
    ),
    f!(
        /// [AVX-VNNI] --- Advanced Vector Extensions - Vector Neural Network Instructions
        ///
        /// [AVX-VNNI]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA
        struct avx::Avxvnni("AVX-VNNI"): "avxvnni" + ["avx2"]
        fn uses_avxvnni
    ),
    f!(
        /// [AVX-VNNI-INT16] --- Advanced Vector Extensions - Vector Neural Network Instructions with 16-bit Integers
        ///
        /// [AVX-VNNI-INT16]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA
        struct avx::Avxvnniint16("AVX-VNNI-INT16"): "avxvnniint16" + ["avx2"]
        fn uses_avxvnniint16
    ),
    f!(
        /// [AVX-VNNI-INT8] --- Advanced Vector Extensions - Vector Neural Network Instructions with 8-bit Integers
        ///
        /// [AVX-VNNI-INT8]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA
        struct avx::Avxvnniint8("AVX-VNNI-INT8"): "avxvnniint8" + ["avx2"]
        fn uses_avxvnniint8
    ),
    f!(
        /// [BMI1] --- Bit Manipulation Instruction Sets
        ///
        /// [BMI1]: https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets
        struct v3::Bmi1(" 1"): "bmi1" + []
        fn uses_bmi1
    ),
    f!(
        /// [BMI2] --- Bit Manipulation Instruction Sets 2
        ///
        /// [BMI2]: https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#BMI2
        struct v3::Bmi2("BMI2"): "bmi2" + []
        fn uses_bmi2
    ),
    f!(
        /// ["cmpxchg16b"] --- Compares and exchange 16 bytes (128 bits) of data atomically
        ///
        /// ["cmpxchg16b"]: https://www.felixcloutier.com/x86/cmpxchg8b:cmpxchg16b
        struct v2::Cmpxchg16b("`cmpxchg16b`"): "cmpxchg16b" + []
        fn uses_cmpxchg16b
    ),
    f!(
        /// [F16C] --- 16-bit floating point conversion instructions
        ///
        /// [F16C]: https://en.wikipedia.org/wiki/F16C
        struct v3::F16c("F16C"): "f16c" + ["avx"]
        fn uses_f16c
    ),
    f!(
        /// [FMA3] --- Three-operand fused multiply-add
        ///
        /// [FMA3]: https://en.wikipedia.org/wiki/FMA_instruction_set
        struct v3::Fma("FMA3"): "fma" + ["avx"]
        fn uses_fma
    ),
    f!(
        /// ["fxsave"] and ["fxrstor"] --- Save and restore x87 FPU, MMX Technology, and SSE State
        ///
        /// ["fxsave"]: https://www.felixcloutier.com/x86/fxsave,
        /// ["fxrstor"]: https://www.felixcloutier.com/x86/fxrstor,
        struct sse::Fxsr("`fxsave + fxrstor`"): "fxsr" + []
        fn uses_fxsr
    ),
    f!(
        /// [GFNI] --- Galois Field New Instructions
        ///
        /// [GFNI]: https://en.wikipedia.org/wiki/AVX-512#GFNI
        struct crypto::Gfni("GFNI"): "gfni" + ["sse2"]
        fn uses_gfni
    ),
    f!(
        /// [KEYLOCKER] --- Intel Key Locker Instructions
        ///
        /// [KEYLOCKER]: https://en.wikipedia.org/wiki/List_of_x86_cryptographic_instructions#Intel_Key_Locker_instructions
        struct crypto::Keylocker("KEYLOCKER"): "kl" + []
        fn uses_keylocker
    ),
    f!(
        /// ["lzcnt"] --- Leading zeros count
        ///
        /// ["lzcnt"]: https://www.felixcloutier.com/x86/lzcnt
        struct v3::Lzcnt("`lzcnt`"): "lzcnt" + []
        fn uses_lzcnt
    ),
    f!(
        /// ["movbe"] --- Move data after swapping bytes
        ///
        /// ["movbe"]: https://www.felixcloutier.com/x86/movbe
        struct v3::Movbe("`movbe`"): "movbe" + []
        fn uses_movbe
    ),
    f!(
        /// ["pclmulqdq"] --- Packed carry-less multiplication quadword
        ///
        /// ["pclmulqdq"]: https://www.felixcloutier.com/x86/pclmulqdq
        struct crypto::Pclmulqdq("`pclmulqdq`"): "pclmulqdq" + ["sse2"]
        fn uses_pclmulqdq
    ),
    f!(
        /// ["popcnt"] --- Count of bits set to 1
        ///
        /// ["popcnt"]: https://www.felixcloutier.com/x86/popcnt
        struct v2::Popcnt("`popcnt`"): "popcnt" + []
        fn uses_popcnt
    ),
    f!(
        /// ["rdrand"] --- Read random number
        ///
        /// ["rdrand"]: https://en.wikipedia.org/wiki/RdRand
        struct crypto::Rdrand("`rdrand`"): "rdrand" + []
        fn uses_rdrand
    ),
    f!(
        /// ["rdseed"] --- Read random seed
        ///
        /// ["rdseed"]: https://en.wikipedia.org/wiki/RdRand
        struct crypto::Rdseed("`rdseed`"): "rdseed" + []
        fn uses_rdseed
    ),
    f!(
        /// [SHA] --- Secure Hash Algorithm
        ///
        /// [SHA]: https://en.wikipedia.org/wiki/Intel_SHA_extensions
        struct crypto::Sha("SHA"): "sha" + ["sse2"]
        fn uses_sha
    ),
    f!(
        /// [SHA512] --- Secure Hash Algorithm with 512-bit digest
        ///
        /// [SHA512]: https://en.wikipedia.org/wiki/Intel_SHA_extensions
        struct crypto::Sha512("SHA512"): "sha512" + ["avx2"]
        fn uses_sha512
    ),
    f!(
        /// [SM3] --- ShangMi 3 Hash Algorithm
        ///
        /// [SM3]: https://en.wikipedia.org/wiki/List_of_x86_cryptographic_instructions#Intel_SHA_and_SM3_instructions
        struct crypto::Sm3("SM3"): "sm3" + ["avx"]
        fn uses_sm3
    ),
    f!(
        /// [SM4] --- ShangMi 4 Cipher Algorithm
        ///
        /// [SM4]: https://en.wikipedia.org/wiki/List_of_x86_cryptographic_instructions#Intel_SHA_and_SM3_instructions
        struct crypto::Sm4("SM4"): "sm4" + ["avx2"]
        fn uses_sm4
    ),
    f!(
        /// [SSE] --- Streaming <abbr title="Single Instruction Multiple Data">SIMD</abbr> Extensions
        ///
        /// [SSE]: https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions
        struct sse::Sse("SSE"): "sse"  + []
        fn uses_sse
    ),
    f!(
        /// [SSE2] --- Streaming <abbr title="Single Instruction Multiple Data">SIMD</abbr> Extensions 2
        ///
        /// [SSE2]: https://en.wikipedia.org/wiki/SSE2
        struct sse::Sse2("SSE2"): "sse2" + ["sse"]
        fn uses_sse2
    ),
    f!(
        /// [SSE3] --- Streaming <abbr title="Single Instruction Multiple Data">SIMD</abbr> Extensions 3
        ///
        /// [SSE3]: https://en.wikipedia.org/wiki/SSE3
        struct sse::Sse3("SSE3"): "sse3" + ["sse2"]
        fn uses_sse3
    ),
    f!(
        /// [SSE4.1] --- Streaming <abbr title="Single Instruction Multiple Data">SIMD</abbr> Extensions 4.1
        ///
        /// [SSE4.1]: https://en.wikipedia.org/wiki/SSE4#SSE4.1
        struct sse::Sse4_1("SSE4.1"): "sse4.1" + ["ssse3"]
        fn uses_sse4
    ),
    f!(
        /// [SSE4.2] --- Streaming<abbr title="Single Instruction Multiple Data">SIMD</abbr>Extensions 4.2
        ///
        /// [SSE4.2]: https://en.wikipedia.org/wiki/SSE4#SSE4.2
        struct sse::Sse4_2("SSE4.2"): "sse4.2" + ["sse4.1"]
        fn uses_sse4
    ),
    f!(
        /// [SSE4a] --- Streaming<abbr title="Single Instruction Multiple Data">SIMD</abbr>Extensions 4a
        ///
        /// [SSE4a]: https://en.wikipedia.org/wiki/SSE4#SSE4a
        struct sse::Sse4a("SSE4a"): "sse4a" + ["sse3"]
        fn uses_sse4a
    ),
    f!(
        /// [SSSE3] --- Supplemental Streaming<abbr title="Single Instruction Multiple Data">SIMD</abbr>Extensions 3
        ///
        /// [SSSE3]: https://en.wikipedia.org/wiki/SSSE3
        struct sse::SupplementalSse3("SSSE3"): "ssse3" + ["sse3"]
        fn uses_ssse3
    ),
    f!(
        /// [TBM] --- Trailing Bit Manipulation
        ///
        /// [TBM]: https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set#TBM_(Trailing_Bit_Manipulation)
        struct discontinued::Tbm("TBM"): "tbm" + []
        fn uses_tbm
    ),
    f!(
        /// [VAES] --- Vector AES Instructions
        ///
        /// [VAES]: https://en.wikipedia.org/wiki/AVX-512#VAES
        struct crypto::Vaes("VAES"): "vaes" + ["avx2", "aes"]
        fn uses_vaes
    ),
    f!(
        /// [VPCLMULQDQ] --- Vector Carry-less multiplication of Quadwords
        ///
        /// [VPCLMULQDQ]: https://en.wikipedia.org/wiki/AVX-512#VPCLMULQDQ
        struct crypto::Vpclmulqdq("VPCLMULQDQ"): "vpclmulqdq" + ["avx", "pclmulqdq"]
        fn uses_vpclmulqdq
    ),
    f!(
        /// [KEYLOCKER_WIDE] --- Intel Wide Keylocker Instructions
        ///
        /// [KEYLOCKER_WIDE]: https://en.wikipedia.org/wiki/List_of_x86_cryptographic_instructions#Intel_Key_Locker_instructions
        struct crypto::WideKeylocker("WIDE KEYLOCKER"): "widekl" + ["kl"]
        fn uses_wide_keylocker
    ),
    f!(
        /// [`xsave`] --- Save processor extended states
        ///
        /// ["xsave"]: https://www.felixcloutier.com/x86/xsave
        struct xsave::Xsave("`xsave`"): "xsave" + []
        fn uses_xsave
    ),
    f!(
        /// ["xsavec"] --- Save processor extended states with compaction
        ///
        /// ["xsavec"]: https://www.felixcloutier.com/x86/xsavec
        struct xsave::Xsavec("`xsavec`"): "xsavec" + []
        fn uses_xsavec
    ),
    f!(
        /// ["xsaveopt"] --- Save processor extended states optimized
        ///
        /// ["xsaveopt"]: https://www.felixcloutier.com/x86/xsaveopt
        struct xsave::Xsaveopt("`xsaveopt`"): "xsaveopt" + []
        fn uses_xsaveopt
    ),
    f!(
        /// ["xsaves"] --- Save processor extended states supervisor
        ///
        /// ["xsaves"]: https://www.felixcloutier.com/x86/xsaves
        struct xsave::Xsaves("`xsaves`"): "xsaves" + []
        fn uses_xsaves
    ),
];

// All taken from <https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels>

pub(crate) const X86_LEVEL_TEMPLATE: &str = include_str!("../../templates/x86_level.rs");

/// The target features required in the x86-64-v1 level.
// Rust doesn't have target features for "cmov", "cmpxchg8b", "fpu", "sce", and "mmx".
// The first four are all assumed, and the final is not implemented because
// it's practically impossible to use correctly (and there's no reason to).
pub(crate) const X86_V1: &[&str] = &["fxsr", "sse", "sse2"];
/// The target features required in the x86-64-v1 level, in addition to those already in [`X86_V1`].
pub(crate) const X86_V2: &[&str] = &[
    "sse3",
    "ssse3",
    "sse4.1",
    "sse4.2",
    "popcnt",
    "cmpxchg16b",
    // The lahfahf target feature is currently in Rust beta.
    // "lahfsahf",
];
/// The target features required in the x86-64-v3 level, excluding those already in [`X86_V2`].
pub(crate) const X86_V3: &[&str] = &[
    "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "lzcnt", "movbe", "xsave",
];
/// The target features required in the x86-64-v4 level, excluding those already in [`X86_V3`].
pub(crate) const X86_V4: &[&str] = &["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"];

#![allow(dead_code)]

// Hand-written expansion of an attribute on a SIMD region/module. The raw
// graph stays ordinary Rust; only region entry points choose a CPU backend.
mod raw {
    pub struct Gain(pub u32);

    impl Gain {
        #[inline(always)]
        pub fn apply(&self, value: u32) -> u32 {
            value.wrapping_mul(self.0)
        }

        #[inline(always)]
        pub fn offset(value: u32) -> u32 {
            value.wrapping_add(0x12345678)
        }
    }

    #[inline(always)]
    pub fn helper(value: u32) -> u32 {
        value.rotate_left(7)
    }

    macro_rules! through_macro {
        ($value:expr) => {
            helper($value)
        };
    }

    #[inline(always)]
    pub fn kernel(xs: &mut [u32], gain: &Gain) {
        let local = |x| through_macro!(Gain::offset(gain.apply(x)));
        for x in xs {
            *x = local(*x);
        }
    }
}

pub use raw::Gain;

#[unsafe(no_mangle)]
#[inline(never)]
pub fn region_scalar(xs: &mut [u32], gain: &Gain) {
    raw::kernel(xs, gain)
}

#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn region_avx2(xs: &mut [u32], gain: &Gain) {
    raw::kernel(xs, gain)
}

#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn manual_avx2(xs: &mut [u32], gain: &Gain) {
    for x in xs {
        *x = x
            .wrapping_mul(gain.0)
            .wrapping_add(0x12345678)
            .rotate_left(7);
    }
}

pub fn kernel(xs: &mut [u32], gain: &Gain) {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { region_avx2(xs, gain) }
    } else {
        region_scalar(xs, gain)
    }
}

fn main() {
    let values = [0, 1, 42, u32::MAX, 0x80000000, 0xdeadbeef];
    let mut input = Vec::from_iter(values.into_iter().cycle().take(1025));
    let mut expected = input.clone();
    let gain = Gain(33);
    region_scalar(&mut expected, &gain);
    kernel(&mut input, &gain);
    assert_eq!(input, expected);
    if std::is_x86_feature_detected!("avx2") {
        let mut direct = Vec::from_iter(values.into_iter().cycle().take(1025));
        unsafe {
            manual_avx2(&mut direct, &gain);
        }
        assert_eq!(input, direct);
    }
    println!("region/method/associated/macro/local-callable correctness passed");
}

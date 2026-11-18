#![allow(non_upper_case_globals)]
#![deny(unsafe_op_in_unsafe_fn)]
use core::marker::PhantomData;
use core::ops::Deref;

mod backend {
    /// Evidence that the current CPU supports AVX2. The field is private.
    #[derive(Clone, Copy)]
    pub struct Avx2(());

    impl Avx2 {
        /// # Safety
        /// The caller must establish that AVX2 is available.
        #[inline(always)]
        pub unsafe fn assume_enabled() -> Self {
            Self(())
        }
    }
}
pub use backend::Avx2;

// Public facade remains callable with foo(x). Unlike a fn item its concrete
// type is nameable, so it can participate in a backend protocol.
#[derive(Clone, Copy)]
pub struct Mix;
pub const mix: Mix = Mix;
impl Deref for Mix {
    type Target = fn(u32) -> u32;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &(dispatch_mix as fn(u32) -> u32)
    }
}

#[inline(always)]
fn raw_mix(x: u32) -> u32 {
    x.wrapping_mul(0x9e3779b9).rotate_left(7)
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_mix(x: u32) -> u32 {
    raw_mix(x)
}

fn dispatch_mix(x: u32) -> u32 {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { avx2_mix(x) }
    } else {
        raw_mix(x)
    }
}

// Autoref dispatch plus a type-only probe preserves the call expression's
// ordinary Fn/FnMut/FnOnce borrowing. The ordinary branch invokes a closure
// containing the original call, instead of forcing the callable to move.
pub struct Probe<F>(PhantomData<fn() -> F>);
pub fn probe<F>(_: &F) -> Probe<F> {
    Probe(PhantomData)
}

pub trait Specialized<A> {
    type Output;
    fn invoke(witness: Avx2, arg: A) -> Self::Output;
}
impl Specialized<u32> for Mix {
    type Output = u32;
    #[inline(always)]
    fn invoke(_: Avx2, arg: u32) -> u32 {
        // Possession of the privately constructed witness establishes AVX2.
        unsafe { avx2_mix(arg) }
    }
}

pub trait ProbeCall<A, R> {
    fn probe_call(self, witness: Avx2, arg: A, fallback: impl FnOnce(A) -> R) -> R;
}
impl<F, A, R> ProbeCall<A, R> for &Probe<F> {
    #[inline(always)]
    fn probe_call(self, _: Avx2, arg: A, fallback: impl FnOnce(A) -> R) -> R {
        fallback(arg)
    }
}
impl<F: Specialized<A, Output = R>, A, R> ProbeCall<A, R> for &&Probe<F> {
    #[inline(always)]
    fn probe_call(self, witness: Avx2, arg: A, _: impl FnOnce(A) -> R) -> R {
        F::invoke(witness, arg)
    }
}

// Stands in for a proc-macro expression rewrite. Restricting this to paths
// avoids evaluating an arbitrary side-effecting callee expression twice.
macro_rules! call {
    ($witness:expr, $f:path, $arg:expr) => {{
        let p = probe(&$f);
        (&&p).probe_call($witness, $arg, |arg| $f(arg))
    }};
}

fn ordinary(x: u32) -> u32 {
    x.wrapping_add(5)
}

#[unsafe(no_mangle)]
#[target_feature(enable = "avx2")]
pub fn marker_process(xs: &mut [u32]) {
    // The hypothetical attribute generates this witness inside the established
    // target-feature context, and rewrites f(x) to call!(witness, f, x).
    let witness = unsafe { Avx2::assume_enabled() };
    let f = mix;
    let local = |x: u32| x.wrapping_add(11);
    for x in xs {
        *x = call!(witness, f, *x);
        *x = call!(witness, ordinary, *x);
        *x = call!(witness, local, *x);
        *x = call!(witness, u32::reverse_bits, *x);
    }
}

#[unsafe(no_mangle)]
#[target_feature(enable = "avx2")]
pub unsafe fn manual_process(xs: &mut [u32]) {
    for x in xs {
        *x = raw_mix(*x).wrapping_add(16).reverse_bits();
    }
}

fn main() {
    assert_eq!(mix(7), raw_mix(7));
    if std::is_x86_feature_detected!("avx2") {
        let witness = unsafe { Avx2::assume_enabled() };
        let mut actual: Vec<u32> = (0..1025).collect();
        let mut expected = actual.clone();
        unsafe {
            marker_process(&mut actual);
            manual_process(&mut expected);
        }
        assert_eq!(actual, expected);

        let owned = String::from("owned capture");
        let borrowed = |x: u32| x + owned.len() as u32;
        assert_eq!(call!(witness, borrowed, 1), 14);
        assert_eq!(call!(witness, borrowed, 2), 15);

        let mut state = 0;
        let mut mutable = |x: u32| {
            state += x;
            state
        };
        assert_eq!(call!(witness, mutable, 1), 1);
        assert_eq!(call!(witness, mutable, 2), 3);
        assert_eq!(state, 3);

        let consumed = String::from("consume me");
        let once = |()| {
            drop(consumed);
            42
        };
        assert_eq!(call!(witness, once, ()), 42);
    }
    println!(
        "callable marker, local alias, ordinary function, closure, associated function passed"
    );
}

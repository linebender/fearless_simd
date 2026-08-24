// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported integer vector type.

#[simd_test]
fn reduce_xor_i8x16<S: Simd>(simd: S) {
    let value = i8x16::simd_from(
        simd,
        [
            23, 43, -65, 20, 73, 110, -101, -84, 63, -45, 103, -68, 49, 54, 51, 68,
        ],
    );
    assert_eq!(value.reduce_xor(), -64);
}

#[simd_test]
fn reduce_xor_u8x16<S: Simd>(simd: S) {
    let value = u8x16::simd_from(
        simd,
        [
            23, 43, 191, 20, 73, 110, 155, 172, 63, 211, 103, 188, 49, 54, 51, 68,
        ],
    );
    assert_eq!(value.reduce_xor(), 192);
}

#[simd_test]
fn reduce_xor_i8x32<S: Simd>(simd: S) {
    let value = i8x32::simd_from(
        simd,
        [
            23, 43, -65, 20, 73, 110, -101, -84, -65, -45, 103, -68, 49, 54, 51, 84, -25, 123, 15,
            -28, -103, -34, -21, -4, 15, 35, -73, 12, 65, 102, -125, -76,
        ],
    );
    assert_eq!(value.reduce_xor(), -16);
}

#[simd_test]
fn reduce_xor_u8x32<S: Simd>(simd: S) {
    let value = u8x32::simd_from(
        simd,
        [
            23, 43, 191, 20, 73, 110, 155, 172, 191, 211, 103, 188, 49, 54, 51, 84, 231, 123, 15,
            228, 153, 222, 235, 252, 15, 35, 183, 12, 65, 102, 131, 180,
        ],
    );
    assert_eq!(value.reduce_xor(), 240);
}

#[simd_test]
fn reduce_xor_i8x64<S: Simd>(simd: S) {
    let value = i8x64::simd_from(
        simd,
        [
            23, 43, -65, 20, 73, 110, -101, -84, -65, -45, 103, -68, 49, 54, 51, 84, 103, 123, 15,
            -28, -103, -34, -21, -4, 15, 35, -73, 12, 65, 102, -125, -92, 55, -53, 95, -76, 41, 14,
            59, 76, 95, 115, 7, -36, -111, -42, -45, -12, 7, 27, -81, 4, 121, 126, -117, -100, -81,
            -61, 87, -84, 33, 6, 35, 84,
        ],
    );
    assert_eq!(value.reduce_xor(), -48);
}

#[simd_test]
fn reduce_xor_u8x64<S: Simd>(simd: S) {
    let value = u8x64::simd_from(
        simd,
        [
            23, 43, 191, 20, 73, 110, 155, 172, 191, 211, 103, 188, 49, 54, 51, 84, 103, 123, 15,
            228, 153, 222, 235, 252, 15, 35, 183, 12, 65, 102, 131, 164, 55, 203, 95, 180, 41, 14,
            59, 76, 95, 115, 7, 220, 145, 214, 211, 244, 7, 27, 175, 4, 121, 126, 139, 156, 175,
            195, 87, 172, 33, 6, 35, 84,
        ],
    );
    assert_eq!(value.reduce_xor(), 208);
}

#[simd_test]
fn reduce_xor_i16x8<S: Simd>(simd: S) {
    let value = i16x8::simd_from(
        simd,
        [31767, -1750, -3009, -4076, -13207, -6034, 27795, -7764],
    );
    assert_eq!(value.reduce_xor(), 10286);
}

#[simd_test]
fn reduce_xor_u16x8<S: Simd>(simd: S) {
    let value = u16x8::simd_from(
        simd,
        [31767, 63786, 62527, 61460, 52329, 59502, 27795, 57772],
    );
    assert_eq!(value.reduce_xor(), 10286);
}

#[simd_test]
fn reduce_xor_i16x16<S: Simd>(simd: S) {
    let value = i16x16::simd_from(
        simd,
        [
            31767, -1750, -3009, -4076, 19561, -6034, 27795, -8020, -8515, -10029, 21607, -28420,
            19761, -9946, 17715, -15280,
        ],
    );
    assert_eq!(value.reduce_xor(), 32431);
}

#[simd_test]
fn reduce_xor_u16x16<S: Simd>(simd: S) {
    let value = u16x16::simd_from(
        simd,
        [
            31767, 63786, 62527, 61460, 19561, 59502, 27795, 57516, 57021, 55507, 21607, 37116,
            19761, 55590, 17715, 50256,
        ],
    );
    assert_eq!(value.reduce_xor(), 32431);
}

#[simd_test]
fn reduce_xor_i16x32<S: Simd>(simd: S) {
    let value = i16x32::simd_from(
        simd,
        [
            31767, -1750, -3009, -4076, 19561, -6034, 27795, -8020, 24253, -10029, 21607, -28420,
            19761, -9946, 17715, -15024, -17049, -18310, -19057, -19996, 3513, -22050, 11747,
            -24068, 7181, -26077, 5815, -11700, 3649, -25994, 1667, -30816,
        ],
    );
    assert_eq!(value.reduce_xor(), -32416);
}

#[simd_test]
fn reduce_xor_u16x32<S: Simd>(simd: S) {
    let value = u16x32::simd_from(
        simd,
        [
            31767, 63786, 62527, 61460, 19561, 59502, 27795, 57516, 24253, 55507, 21607, 37116,
            19761, 55590, 17715, 50512, 48487, 47226, 46479, 45540, 3513, 43486, 11747, 41468,
            7181, 39459, 5815, 53836, 3649, 39542, 1667, 34720,
        ],
    );
    assert_eq!(value.reduce_xor(), 33120);
}

#[simd_test]
fn reduce_xor_i32x4<S: Simd>(simd: S) {
    let value = i32x4::simd_from(simd, [2135587863, -23791318, -35654593, -43454380]);
    assert_eq!(value.reduce_xor(), -2123792042);
}

#[simd_test]
fn reduce_xor_u32x4<S: Simd>(simd: S) {
    let value = u32x4::simd_from(simd, [2135587863, 4271175978, 4259312703, 4251512916]);
    assert_eq!(value.reduce_xor(), 2171175254);
}

#[simd_test]
fn reduce_xor_i32x8<S: Simd>(simd: S) {
    let value = i32x8::simd_from(
        simd,
        [
            2135587863, -23791318, 2111829055, -43388844, -596349847, -71374738, 2064215187,
            -94969688,
        ],
    );
    assert_eq!(value.reduce_xor(), -1476556438);
}

#[simd_test]
fn reduce_xor_u32x8<S: Simd>(simd: S) {
    let value = u32x8::simd_from(
        simd,
        [
            2135587863, 4271175978, 2111829055, 4251578452, 3698617449, 4223592558, 2064215187,
            4199997608,
        ],
    );
    assert_eq!(value.reduce_xor(), 2818410858);
}

#[simd_test]
fn reduce_xor_i32x16<S: Simd>(simd: S) {
    let value = i32x16::simd_from(
        simd,
        [
            2135587863, -23791318, 2111829055, -43388844, 1551133801, -71374738, 2064215187,
            -94904152, -73507651, -118957869, 2016629863, -142765828, 1994935569, -434976474,
            1969046835, -190397104,
        ],
    );
    assert_eq!(value.reduce_xor(), -1161441845);
}

#[simd_test]
fn reduce_xor_u32x16<S: Simd>(simd: S) {
    let value = u32x16::simd_from(
        simd,
        [
            2135587863, 4271175978, 2111829055, 4251578452, 1551133801, 4223592558, 2064215187,
            4200063144, 4221459645, 4176009427, 2016629863, 4152201468, 1994935569, 3859990822,
            1969046835, 4104570192,
        ],
    );
    assert_eq!(value.reduce_xor(), 3133525451);
}

#[simd_test]
fn reduce_xor_i64x2<S: Simd>(simd: S) {
    let value = i64x2::simd_from(simd, [-7046029254386353129, -4868686467622962902]);
    assert_eq!(value.reduce_xor(), 2475162072583669053);
}

#[simd_test]
fn reduce_xor_u64x2<S: Simd>(simd: S) {
    let value = u64x2::simd_from(simd, [11400714819323198487, 13578057606086588714]);
    assert_eq!(value.reduce_xor(), 2475162072583669053);
}

#[simd_test]
fn reduce_xor_i64x4<S: Simd>(simd: S) {
    let value = i64x4::simd_from(
        simd,
        [
            -7046029254386353129,
            4354685564936845610,
            6532028347405300799,
            8709371125582917716,
        ],
    );
    assert_eq!(value.reduce_xor(), -9213800775226457770);
}

#[simd_test]
fn reduce_xor_u64x4<S: Simd>(simd: S) {
    let value = u64x4::simd_from(
        simd,
        [
            11400714819323198487,
            4354685564936845610,
            6532028347405300799,
            8709371125582917716,
        ],
    );
    assert_eq!(value.reduce_xor(), 9232943298483093846);
}

#[simd_test]
fn reduce_xor_i64x8<S: Simd>(simd: S) {
    let value = i64x8::simd_from(
        simd,
        [
            -7046029254386353129,
            4354685564936845610,
            -2691343689449475009,
            8709371129877885012,
            -7560030161904309143,
            -5382687447618492290,
            6018036236517205139,
            -1026875918350294872,
        ],
    );
    assert_eq!(value.reduce_xor(), -9205311402351754886);
}

#[simd_test]
fn reduce_xor_u64x8<S: Simd>(simd: S) {
    let value = u64x8::simd_from(
        simd,
        [
            11400714819323198487,
            4354685564936845610,
            15755400384260076607,
            8709371129877885012,
            10886713911805242473,
            13064056626091059326,
            6018036236517205139,
            17419868155359256744,
        ],
    );
    assert_eq!(value.reduce_xor(), 9241432671357796730);
}

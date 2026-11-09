use simd_research_macros::simd;

fn increment(value: &mut u32) {
    *value += 1;
}

#[simd]
fn mutable_reborrows(value: &mut u32) {
    increment(value);
    increment(value);
}

#[test]
fn ordinary_calls_preserve_implicit_mutable_reborrows() {
    let mut value = 0;
    mutable_reborrows(&mut value);
    assert_eq!(value, 2);
}

#[simd]
fn annotated_increment(value: &mut u32) {
    *value += 1;
}

#[simd]
fn annotated_mutable_reborrows(value: &mut u32) {
    annotated_increment(value);
    annotated_increment(value);
}

#[test]
fn specialized_calls_preserve_implicit_mutable_reborrows() {
    let mut value = 0;
    annotated_mutable_reborrows(&mut value);
    assert_eq!(value, 2);
}

fn slice_length(values: &[u32]) -> usize {
    values.len()
}

fn apply_pointer(function: fn(u32) -> u32) -> u32 {
    function(4)
}

fn double(value: u32) -> u32 {
    value * 2
}

#[simd]
fn argument_coercions(values: &Vec<u32>) -> (usize, u32) {
    (slice_length(values), apply_pointer(double))
}

#[test]
fn ordinary_calls_preserve_deref_and_function_pointer_coercions() {
    assert_eq!(argument_coercions(&vec![1, 2, 3]), (3, 8));
}

fn identity(value: &str) -> &str {
    value
}

#[simd]
fn borrowed_return(value: &str) -> &str {
    identity(value)
}

#[test]
fn ordinary_calls_can_return_borrowed_arguments() {
    let value = String::from("borrowed");
    assert_eq!(borrowed_return(&value), "borrowed");
}

#[simd]
fn annotated_identity(value: &str) -> &str {
    value
}

#[simd]
fn annotated_borrowed_return(value: &str) -> &str {
    annotated_identity(value)
}

#[test]
fn specialized_calls_can_return_borrowed_arguments() {
    let value = String::from("borrowed");
    assert_eq!(annotated_borrowed_return(&value), "borrowed");
}

#[simd]
fn callable(value: u32) -> u32 {
    value + 10
}

#[simd]
fn callable_parameter(callable: fn(u32) -> u32) -> u32 {
    callable(2)
}

#[simd]
fn callable_local() -> u32 {
    let callable = |value: u32| value + 100;
    callable(2)
}

#[simd]
fn callable_nested_scope() -> u32 {
    let first = callable(2);
    let second = {
        let callable = |value: u32| value + 100;
        callable(2)
    };
    first + second
}

#[test]
fn local_callable_shadowing_preserves_the_original_binding() {
    assert_eq!(callable_parameter(double), 4);
    assert_eq!(callable_local(), 102);
    assert_eq!(callable_nested_scope(), 114);
}

fn mark(log: &mut Vec<u32>, value: u32) -> u32 {
    log.push(value);
    value
}

fn sum(a: u32, b: u32) -> u32 {
    a + b
}

#[simd]
fn evaluation_order(log: &mut Vec<u32>) -> u32 {
    sum(mark(log, 1), mark(log, 2))
}

#[test]
fn original_arguments_are_evaluated_once_in_order() {
    let mut log = Vec::new();
    assert_eq!(evaluation_order(&mut log), 3);
    assert_eq!(log, [1, 2]);
}

#[simd]
fn argument_control_flow(value: Result<u32, ()>) -> Result<u32, ()> {
    let result = double(value?);
    Ok(result)
}

#[test]
fn question_mark_keeps_the_original_function_boundary() {
    assert_eq!(argument_control_flow(Ok(3)), Ok(6));
    assert_eq!(argument_control_flow(Err(())), Err(()));
}

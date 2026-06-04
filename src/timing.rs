/// Run a closure and return (result, elapsed wall-clock time).
pub fn timed<T, F: FnOnce() -> T>(f: F) -> (T, std::time::Duration) {
    let start = std::time::Instant::now();
    let result = f();
    (result, start.elapsed())
}

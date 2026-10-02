//! Emit our generated model action map for an external format comparison.
fn main() {
    for (index, slot) in rz_encoding::policy::slots().iter().enumerate() {
        let square = |s: u8| format!("{}{}", char::from(b'a' + s % 8), 1 + s / 8);
        let suffix = slot.promotion.map(|p| p.to_string()).unwrap_or_default();
        println!("{index}\t{}{}{suffix}", square(slot.from), square(slot.to));
    }
}

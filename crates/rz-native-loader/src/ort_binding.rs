//! Synchronous ORT binding Run without acquiring duplicate native output values.
//! Fixed outputs are already owned by the caller. ort rc.10's run_binding clones
//! its Rust fixed values but does not release GetBoundOutputValues' native copies.
use ort::{io_binding::IoBinding, session::Session, sys, AsPointer};
use std::ffi::c_char;

const MAX_MESSAGE: usize = 1024;
type GetCode = unsafe extern "system" fn(*const sys::OrtStatus) -> sys::OrtErrorCode;
type GetMessage = unsafe extern "system" fn(*const sys::OrtStatus) -> *const c_char;
type Release = unsafe extern "system" fn(*mut sys::OrtStatus);

struct StatusOwner {
    status: *mut sys::OrtStatus,
    release: Release,
}
impl Drop for StatusOwner {
    fn drop(&mut self) {
        // SAFETY: the non-null native status is owned exactly once here and
        // remains valid until this matching API table's ReleaseStatus call.
        unsafe { (self.release)(self.status) };
    }
}

/// Uses the same initialized, process-pinned ORT API as these typed handles.
/// Requires exclusive session access. No asynchronous/terminate RunOptions are
/// set. Success completes the synchronous Run; error does not attest completion.
/// The caller still owns all bound tensors and performs its normal output fence.
pub fn run_fixed_binding(session: &mut Session, binding: &IoBinding) -> ort::Result<()> {
    let api = ort::api();
    // SAFETY: ort owns both non-null handles and pins their native session. This
    // is the same synchronous call/signature as Session::run_binding, stopping
    // before GetBoundOutputValues because output tensors are already owned.
    let status =
        unsafe { (api.RunWithBinding)(session.ptr_mut(), std::ptr::null(), binding.ptr()) };
    status_result(
        status.0,
        api.GetErrorCode,
        api.GetErrorMessage,
        api.ReleaseStatus,
    )
}

fn status_result(
    status: *mut sys::OrtStatus,
    code: GetCode,
    message: GetMessage,
    release: Release,
) -> ort::Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let owner = StatusOwner { status, release };
    // SAFETY: an owned native status is valid for these same API accessors. Its
    // message remains native-owned until the guard is dropped after the copy.
    let (code, message) = unsafe { (code(owner.status), message(owner.status)) };
    let mut bytes = Vec::new();
    if !message.is_null() {
        for index in 0..MAX_MESSAGE {
            // SAFETY: native ORT supplies a NUL-terminated status string. Read
            // only until NUL or our bound; no scan through unbounded diagnostics.
            let byte = unsafe { *message.add(index) } as u8;
            if byte == 0 {
                break;
            }
            bytes.push(byte);
        }
    }
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    let truncated = bytes.len() == MAX_MESSAGE || text.len() > MAX_MESSAGE;
    if text.len() > MAX_MESSAGE {
        let mut boundary = MAX_MESSAGE;
        while !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        text.truncate(boundary);
    }
    if text.is_empty() {
        text.push_str("native ORT error without message");
    } else if truncated {
        text.push_str(" [truncated]");
    }
    Err(ort::Error::new_with_code(code.into(), text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        ffi::CString,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };
    struct FakeStatus {
        code: sys::OrtErrorCode,
        message: CString,
        releases: Arc<AtomicUsize>,
    }
    unsafe extern "system" fn get_code(status: *const sys::OrtStatus) -> sys::OrtErrorCode {
        // SAFETY: tests pass only live boxed FakeStatus pointers to these mocks.
        unsafe { (*status.cast::<FakeStatus>()).code }
    }
    unsafe extern "system" fn get_message(status: *const sys::OrtStatus) -> *const c_char {
        // SAFETY: same fixture contract; CString lives until release.
        unsafe { (*status.cast::<FakeStatus>()).message.as_ptr() }
    }
    unsafe extern "system" fn release(status: *mut sys::OrtStatus) {
        // SAFETY: sole owned pointer, allocated by the fixture with Box::into_raw.
        let owned = unsafe { Box::from_raw(status.cast::<FakeStatus>()) };
        owned.releases.fetch_add(1, Ordering::Relaxed);
    }
    #[test]
    fn native_failure_preserves_code_copies_message_and_releases_once() {
        for message in [
            b"original native error".to_vec(),
            vec![b'x'; MAX_MESSAGE + 20],
            vec![255],
            vec![255; MAX_MESSAGE + 20],
        ] {
            let releases = Arc::new(AtomicUsize::new(0));
            let value = Box::new(FakeStatus {
                code: sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
                message: CString::new(message.clone()).unwrap(),
                releases: Arc::clone(&releases),
            });
            let status = Box::into_raw(value).cast::<sys::OrtStatus>();
            let error = status_result(status, get_code, get_message, release).unwrap_err();
            assert_eq!(error.code(), ort::ErrorCode::InvalidArgument);
            assert_eq!(releases.load(Ordering::Relaxed), 1);
            assert!(error.message().len() <= MAX_MESSAGE + 12);
            if message.len() > MAX_MESSAGE {
                assert!(error.message().ends_with(" [truncated]"));
            } else {
                assert_eq!(error.message(), String::from_utf8_lossy(&message));
            }
        }
    }
    #[test]
    fn success_has_no_status_to_read_or_release() {
        status_result(std::ptr::null_mut(), get_code, get_message, release).unwrap();
    }
}

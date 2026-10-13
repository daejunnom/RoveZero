//! 보유 중인 파일 객체에서 원본 커서를 바꾸지 않는 읽기 핸들을 만든다.
//!
//! 이 경계는 파일 경로를 다시 조회하지 않으며 내용의 불변성이나 hash 검증을
//! 대신하지 않는다. 호출자는 반환된 핸들로 읽은 bytes와 원본 metadata를 검증한다.

/// 같은 파일에 대해 독립 커서를 가진 읽기 전용 동기 핸들을 반환한다.
///
/// Windows의 `seek_read`는 커서를 변경하고 `try_clone`는 커서를 공유한다.
/// 이 함수의 결과는 각 읽기 호출이 지역적으로 소유하며, 원본 핸들은 유지된다.
/// 기존 열린 핸들의 접근·공유 조건과 충돌하면 원래 OS 오류를 반환한다.
#[cfg(windows)]
#[allow(unsafe_code)] // 아래 retained-handle FFI 경계에만 적용한다.
pub fn independent_read_handle(file: &std::fs::File) -> std::io::Result<std::fs::File> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, RawHandle};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn ReOpenFile(
            original: RawHandle,
            desired_access: u32,
            share_mode: u32,
            flags: u32,
        ) -> RawHandle;
    }

    const GENERIC_READ: u32 = 0x8000_0000;
    const SHARE_READ_WRITE_DELETE: u32 = 0x0000_0007;
    // SAFETY: `file`이 호출 동안 살아 있는 원본 핸들을 소유한다. 경로를
    // 재조회하지 않고 읽기 권한만 요청하며, flags=0으로 동기 I/O를 유지한다.
    // 공유 flags는 새 쓰기 권한이 아니며 원본의 기존 공유 제한을 완화하지 않는다.
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            GENERIC_READ,
            SHARE_READ_WRITE_DELETE,
            0,
        )
    };
    if handle == (-1isize) as RawHandle {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: 성공한 ReOpenFile이 새 유효 핸들의 소유권을 넘겼다. 이 File만
    // 그 핸들을 소유하며, 이후 읽기나 검증이 실패해도 RAII로 한 번 닫힌다.
    Ok(unsafe { std::fs::File::from_raw_handle(handle) })
}

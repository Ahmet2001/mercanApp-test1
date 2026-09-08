use std::cell::RefCell;
use std::slice;
use std::sync::OnceLock;

use nedo_tokenizer::{SurfaceVocabulary, Tokenizer, TokenizerConfig, TokenizerMode};

const VOCAB: &[u8] = include_bytes!("../../NedoTokenizer/assets/surface-vocab.bin");
const VOCAB_SHA256: &str = "72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600";

struct State {
    tokenizer: Tokenizer<'static>,
    vocabulary: SurfaceVocabulary,
}

static STATE: OnceLock<Result<State, String>> = OnceLock::new();

thread_local! {
    static OUTPUT_IDS: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

fn state() -> Result<&'static State, &'static str> {
    match STATE.get_or_init(|| {
        let tokenizer = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Auto,
            max_sentence_tokens: 512,
            max_fallback_chars: 48,
            contextual_disambiguation: true,
            detect_unmarked_code: true,
        })
        .map_err(|e| format!("NDSRF004 tokenizer init failed: {e}"))?;
        let vocabulary = SurfaceVocabulary::from_bytes(VOCAB)
            .map_err(|e| format!("NDSRF004 vocab init failed: {e}"))?;
        if vocabulary.len() != 32_000 {
            return Err(format!("NDSRF004 vocab size mismatch: {}", vocabulary.len()));
        }
        Ok(State { tokenizer, vocabulary })
    }) {
        Ok(s) => Ok(s),
        Err(e) => {
            eprintln!("{e}");
            Err("NDSRF004 state initialization failed")
        }
    }
}

#[no_mangle]
pub extern "C" fn nedo004_vocab_sha256() -> *const std::ffi::c_char {
    static SHA_C: &[u8] = b"72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600\0";
    SHA_C.as_ptr().cast()
}

#[no_mangle]
pub unsafe extern "C" fn nedo004_encode(
    data: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *const u16 {
    if out_len.is_null() || (len != 0 && data.is_null()) {
        return std::ptr::null();
    }
    *out_len = 0;

    let input: &[u8] = if len == 0 { &[] } else { slice::from_raw_parts(data, len) };
    let st = match state() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let document = match st.tokenizer.tokenize(input.to_vec()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("NDSRF004 tokenize failed: {e}");
            return std::ptr::null();
        }
    };
    let encoded = match st.vocabulary.encode_document(&document, false) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("NDSRF004 surface encode failed: {e}");
            return std::ptr::null();
        }
    };
    if encoded.ids.len() < 2 {
        eprintln!("NDSRF004 encoded document lacks BOS/EOS boundaries");
        return std::ptr::null();
    }

    OUTPUT_IDS.with(|cell| {
        let mut ids = cell.borrow_mut();
        ids.clear();
        ids.extend_from_slice(&encoded.ids[1..encoded.ids.len() - 1]);
        *out_len = ids.len();
        if ids.is_empty() {
            std::ptr::NonNull::<u16>::dangling().as_ptr()
        } else {
            ids.as_ptr()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_vocab_contract_loads() {
        let s = state().unwrap();
        assert_eq!(s.vocabulary.len(), 32_000);
        assert_eq!(VOCAB_SHA256.len(), 64);
    }
}

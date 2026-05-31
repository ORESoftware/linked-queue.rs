//! Coverage-guided differential fuzz target.
//!
//! libFuzzer feeds us bytes; `arbitrary` decodes them into a sequence of
//! operations. We run each op against both a `LinkedQueue` and a `VecDeque`
//! oracle and assert (via `panic!`) that they stay equivalent in return values
//! and in full iteration order. Any divergence is a crash that libFuzzer
//! minimizes for us.
//!
//! Run with:  `cargo +nightly fuzz run queue_ops`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use linked_queue::{InsertError, LinkedQueue};
use std::collections::VecDeque;

type K = u8;
type V = u16;

#[derive(Arbitrary, Debug)]
enum Op {
    PushBack(K, V),
    PushFront(K, V),
    TryPushBack(K, V),
    TryPushFront(K, V),
    PopFront,
    PopBack,
    Remove(K),
    MoveToBack(K),
    MoveToFront(K),
    InsertAfterFront(K, V),
    InsertBeforeBack(K, V),
    GetMutSet(K, V),
    Probe(K),
}

/// Squeeze keys into a small domain so collisions are frequent.
fn mask(k: K) -> K {
    k % 16
}

fn model_pos(m: &VecDeque<(K, V)>, k: K) -> Option<usize> {
    m.iter().position(|(mk, _)| *mk == k)
}

fn model_remove(m: &mut VecDeque<(K, V)>, k: K) -> Option<V> {
    model_pos(m, k).map(|p| m.remove(p).unwrap().1)
}

fn apply(q: &mut LinkedQueue<K, V>, m: &mut VecDeque<(K, V)>, op: Op) {
    match op {
        Op::PushBack(k, v) => {
            let k = mask(k);
            let old = q.push_back(k, v);
            let model_old = model_remove(m, k);
            m.push_back((k, v));
            assert_eq!(old, model_old);
        }
        Op::PushFront(k, v) => {
            let k = mask(k);
            let old = q.push_front(k, v);
            let model_old = model_remove(m, k);
            m.push_front((k, v));
            assert_eq!(old, model_old);
        }
        Op::TryPushBack(k, v) => {
            let k = mask(k);
            let r = q.try_push_back(k, v);
            if model_pos(m, k).is_some() {
                assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
            } else {
                assert!(r.is_ok());
                m.push_back((k, v));
            }
        }
        Op::TryPushFront(k, v) => {
            let k = mask(k);
            let r = q.try_push_front(k, v);
            if model_pos(m, k).is_some() {
                assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
            } else {
                assert!(r.is_ok());
                m.push_front((k, v));
            }
        }
        Op::PopFront => assert_eq!(q.pop_front(), m.pop_front()),
        Op::PopBack => assert_eq!(q.pop_back(), m.pop_back()),
        Op::Remove(k) => {
            let k = mask(k);
            assert_eq!(q.remove(&k), model_remove(m, k));
        }
        Op::MoveToBack(k) => {
            let k = mask(k);
            let moved = q.move_to_back(&k);
            match model_remove(m, k) {
                Some(val) => {
                    assert!(moved);
                    m.push_back((k, val));
                }
                None => assert!(!moved),
            }
        }
        Op::MoveToFront(k) => {
            let k = mask(k);
            let moved = q.move_to_front(&k);
            match model_remove(m, k) {
                Some(val) => {
                    assert!(moved);
                    m.push_front((k, val));
                }
                None => assert!(!moved),
            }
        }
        Op::InsertAfterFront(k, v) => {
            let k = mask(k);
            if let Some((anchor, _)) = m.front().copied() {
                let r = q.insert_after(&anchor, k, v);
                if model_pos(m, k).is_some() {
                    assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
                } else {
                    assert!(r.is_ok());
                    let pos = model_pos(m, anchor).unwrap();
                    m.insert(pos + 1, (k, v));
                }
            }
        }
        Op::InsertBeforeBack(k, v) => {
            let k = mask(k);
            if let Some((anchor, _)) = m.back().copied() {
                let r = q.insert_before(&anchor, k, v);
                if model_pos(m, k).is_some() {
                    assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
                } else {
                    assert!(r.is_ok());
                    let pos = model_pos(m, anchor).unwrap();
                    m.insert(pos, (k, v));
                }
            }
        }
        Op::GetMutSet(k, v) => {
            let k = mask(k);
            match q.get_mut(&k) {
                Some(slot) => {
                    *slot = v;
                    let p = model_pos(m, k).expect("model has the key too");
                    m[p].1 = v;
                }
                None => assert!(model_pos(m, k).is_none()),
            }
        }
        Op::Probe(k) => {
            let k = mask(k);
            assert_eq!(q.contains(&k), model_pos(m, k).is_some());
            assert_eq!(q.get(&k).copied(), m.iter().find(|(mk, _)| *mk == k).map(|(_, v)| *v));
        }
    }
}

fn check(q: &LinkedQueue<K, V>, m: &VecDeque<(K, V)>) {
    assert_eq!(q.len(), m.len());
    let q_fwd: Vec<(K, V)> = q.iter().map(|(k, v)| (*k, *v)).collect();
    let m_fwd: Vec<(K, V)> = m.iter().copied().collect();
    assert_eq!(q_fwd, m_fwd);
    let q_rev: Vec<(K, V)> = q.iter().rev().map(|(k, v)| (*k, *v)).collect();
    let mut m_rev = m_fwd;
    m_rev.reverse();
    assert_eq!(q_rev, m_rev);
    assert_eq!(q.front().map(|(k, v)| (*k, *v)), m.front().copied());
    assert_eq!(q.back().map(|(k, v)| (*k, *v)), m.back().copied());
}

fuzz_target!(|ops: Vec<Op>| {
    let mut q: LinkedQueue<K, V> = LinkedQueue::new();
    let mut m: VecDeque<(K, V)> = VecDeque::new();
    for op in ops {
        apply(&mut q, &mut m, op);
        check(&q, &m);
    }
});

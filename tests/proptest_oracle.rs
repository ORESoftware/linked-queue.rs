//! Property-based differential fuzzing of `LinkedQueue` against a `VecDeque`
//! oracle.
//!
//! Each test case is a randomly generated sequence of operations. We apply
//! every operation to both the real `LinkedQueue` and a simple `VecDeque`
//! model, assert the return values agree, and after *every* step assert the
//! two structures agree on `len`, `is_empty`, full forward iteration order,
//! full reverse iteration order, `front`, and `back`. `proptest` shrinks any
//! failing sequence to a minimal reproducer.
//!
//! Keys are drawn from a deliberately small domain so collisions (and thus the
//! upsert / move / dedup paths) are exercised constantly.

use linked_queue::{InsertError, LinkedQueue};
use proptest::prelude::*;
use std::collections::VecDeque;

type K = u8;
type V = u16;

#[derive(Debug, Clone)]
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
    /// Splice after the current front element (if any).
    InsertAfterFront(K, V),
    /// Splice before the current back element (if any).
    InsertBeforeBack(K, V),
    /// Splice after the element at model position `n % len` — exercises
    /// *middle* anchors, not just the ends.
    InsertAfterNth(usize, K, V),
    /// Splice before the element at model position `n % len`.
    InsertBeforeNth(usize, K, V),
    /// Overwrite a value in place via `get_mut` (must not move the element).
    GetMutSet(K, V),
    /// Read-only probe of `get` + `contains`.
    Probe(K),
}

fn model_pos(m: &VecDeque<(K, V)>, k: K) -> Option<usize> {
    m.iter().position(|(mk, _)| *mk == k)
}

fn model_remove(m: &mut VecDeque<(K, V)>, k: K) -> Option<V> {
    model_pos(m, k).map(|p| m.remove(p).unwrap().1)
}

fn model_get(m: &VecDeque<(K, V)>, k: K) -> Option<V> {
    m.iter().find(|(mk, _)| *mk == k).map(|(_, v)| *v)
}

fn op_strategy() -> impl Strategy<Value = Op> {
    // Small key domain -> frequent collisions.
    let key = 0u8..12u8;
    prop_oneof![
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::PushBack(k, v)),
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::PushFront(k, v)),
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::TryPushBack(k, v)),
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::TryPushFront(k, v)),
        Just(Op::PopFront),
        Just(Op::PopBack),
        key.clone().prop_map(Op::Remove),
        key.clone().prop_map(Op::MoveToBack),
        key.clone().prop_map(Op::MoveToFront),
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::InsertAfterFront(k, v)),
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::InsertBeforeBack(k, v)),
        (any::<usize>(), key.clone(), any::<V>())
            .prop_map(|(n, k, v)| Op::InsertAfterNth(n, k, v)),
        (any::<usize>(), key.clone(), any::<V>())
            .prop_map(|(n, k, v)| Op::InsertBeforeNth(n, k, v)),
        (key.clone(), any::<V>()).prop_map(|(k, v)| Op::GetMutSet(k, v)),
        key.prop_map(Op::Probe),
    ]
}

fn apply(q: &mut LinkedQueue<K, V>, m: &mut VecDeque<(K, V)>, op: Op) -> Result<(), TestCaseError> {
    match op {
        Op::PushBack(k, v) => {
            let old = q.push_back(k, v);
            let model_old = model_remove(m, k);
            m.push_back((k, v));
            prop_assert_eq!(old, model_old, "push_back old value");
        }
        Op::PushFront(k, v) => {
            let old = q.push_front(k, v);
            let model_old = model_remove(m, k);
            m.push_front((k, v));
            prop_assert_eq!(old, model_old, "push_front old value");
        }
        Op::TryPushBack(k, v) => {
            let r = q.try_push_back(k, v);
            if model_pos(m, k).is_some() {
                prop_assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
            } else {
                prop_assert!(r.is_ok());
                m.push_back((k, v));
            }
        }
        Op::TryPushFront(k, v) => {
            let r = q.try_push_front(k, v);
            if model_pos(m, k).is_some() {
                prop_assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
            } else {
                prop_assert!(r.is_ok());
                m.push_front((k, v));
            }
        }
        Op::PopFront => {
            prop_assert_eq!(q.pop_front(), m.pop_front());
        }
        Op::PopBack => {
            prop_assert_eq!(q.pop_back(), m.pop_back());
        }
        Op::Remove(k) => {
            prop_assert_eq!(q.remove(&k), model_remove(m, k));
        }
        Op::MoveToBack(k) => {
            let moved = q.move_to_back(&k);
            match model_remove(m, k) {
                Some(val) => {
                    prop_assert!(moved);
                    m.push_back((k, val));
                }
                None => prop_assert!(!moved),
            }
        }
        Op::MoveToFront(k) => {
            let moved = q.move_to_front(&k);
            match model_remove(m, k) {
                Some(val) => {
                    prop_assert!(moved);
                    m.push_front((k, val));
                }
                None => prop_assert!(!moved),
            }
        }
        Op::InsertAfterFront(k, v) => {
            if let Some((anchor, _)) = m.front().copied() {
                let r = q.insert_after(&anchor, k, v);
                if model_pos(m, k).is_some() {
                    prop_assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
                } else {
                    prop_assert!(r.is_ok());
                    let pos = model_pos(m, anchor).unwrap();
                    m.insert(pos + 1, (k, v));
                }
            }
        }
        Op::InsertBeforeBack(k, v) => {
            if let Some((anchor, _)) = m.back().copied() {
                let r = q.insert_before(&anchor, k, v);
                if model_pos(m, k).is_some() {
                    prop_assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
                } else {
                    prop_assert!(r.is_ok());
                    let pos = model_pos(m, anchor).unwrap();
                    m.insert(pos, (k, v));
                }
            }
        }
        Op::InsertAfterNth(n, k, v) => {
            if !m.is_empty() {
                let pos = n % m.len();
                let anchor = m[pos].0;
                let r = q.insert_after(&anchor, k, v);
                if model_pos(m, k).is_some() {
                    prop_assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
                } else {
                    prop_assert!(r.is_ok());
                    m.insert(pos + 1, (k, v));
                }
            }
        }
        Op::InsertBeforeNth(n, k, v) => {
            if !m.is_empty() {
                let pos = n % m.len();
                let anchor = m[pos].0;
                let r = q.insert_before(&anchor, k, v);
                if model_pos(m, k).is_some() {
                    prop_assert_eq!(r, Err(InsertError::DuplicateKey { key: k, value: v }));
                } else {
                    prop_assert!(r.is_ok());
                    m.insert(pos, (k, v));
                }
            }
        }
        Op::GetMutSet(k, v) => match q.get_mut(&k) {
            Some(slot) => {
                *slot = v;
                let p = model_pos(m, k).expect("model should also have the key");
                m[p].1 = v;
            }
            None => prop_assert!(model_pos(m, k).is_none()),
        },
        Op::Probe(k) => {
            prop_assert_eq!(q.contains(&k), model_pos(m, k).is_some());
            prop_assert_eq!(q.get(&k).copied(), model_get(m, k));
        }
    }
    Ok(())
}

fn check_invariants(q: &LinkedQueue<K, V>, m: &VecDeque<(K, V)>) -> Result<(), TestCaseError> {
    prop_assert_eq!(q.len(), m.len(), "len");
    prop_assert_eq!(q.is_empty(), m.is_empty(), "is_empty");

    let q_fwd: Vec<(K, V)> = q.iter().map(|(k, v)| (*k, *v)).collect();
    let m_fwd: Vec<(K, V)> = m.iter().copied().collect();
    prop_assert_eq!(&q_fwd, &m_fwd, "forward iteration order");

    let q_rev: Vec<(K, V)> = q.iter().rev().map(|(k, v)| (*k, *v)).collect();
    let mut m_rev = m_fwd.clone();
    m_rev.reverse();
    prop_assert_eq!(q_rev, m_rev, "reverse iteration order");

    prop_assert_eq!(
        q.front().map(|(k, v)| (*k, *v)),
        m.front().copied(),
        "front"
    );
    prop_assert_eq!(q.back().map(|(k, v)| (*k, *v)), m.back().copied(), "back");

    // ExactSizeIterator must agree with len.
    prop_assert_eq!(q.iter().len(), m.len(), "iter().len()");

    // Internal structural consistency (head/tail/prev/next, index, free-list)
    // — the part a VecDeque oracle cannot observe.
    q.assert_invariants();
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2048,
        max_shrink_iters: 1_000_000,
        ..ProptestConfig::default()
    })]

    /// Differential test: a random op sequence must keep `LinkedQueue` in
    /// lockstep with a `VecDeque` oracle, step by step.
    #[test]
    fn differential_against_vecdeque(ops in prop::collection::vec(op_strategy(), 0..512)) {
        let mut q: LinkedQueue<K, V> = LinkedQueue::new();
        let mut m: VecDeque<(K, V)> = VecDeque::new();
        for op in ops {
            apply(&mut q, &mut m, op)?;
            check_invariants(&q, &m)?;
        }

        // Clone equivalence and a full drain at the end.
        let c = q.clone();
        prop_assert_eq!(&q, &c);
        let drained: Vec<(K, V)> = q.drain().collect();
        prop_assert_eq!(drained, m.iter().copied().collect::<Vec<_>>());
        prop_assert!(q.is_empty());
    }
}

//! Long-running deterministic stress test.
//!
//! Complements the `proptest` differential test (many short sequences) with a
//! single very long sequence (one million operations) driven by a deterministic
//! LCG. This is good at surfacing issues that only appear after sustained churn
//! — free-list reuse drift, length under/overflow, head/tail corruption — that
//! short randomized cases may not reach. Full equivalence against a `VecDeque`
//! oracle is checked periodically and at the end.

use linked_queue::LinkedQueue;
use std::collections::VecDeque;

type K = u32;
type V = u32;

fn model_remove(m: &mut VecDeque<(K, V)>, k: K) -> Option<V> {
    m.iter()
        .position(|(mk, _)| *mk == k)
        .map(|p| m.remove(p).unwrap().1)
}

fn assert_equiv(q: &LinkedQueue<K, V>, m: &VecDeque<(K, V)>, step: u64) {
    assert_eq!(q.len(), m.len(), "len mismatch at step {step}");
    let q_fwd: Vec<(K, V)> = q.iter().map(|(k, v)| (*k, *v)).collect();
    let m_fwd: Vec<(K, V)> = m.iter().copied().collect();
    assert_eq!(q_fwd, m_fwd, "forward order mismatch at step {step}");
    let q_rev: Vec<(K, V)> = q.iter().rev().map(|(k, v)| (*k, *v)).collect();
    let mut m_rev = m_fwd;
    m_rev.reverse();
    assert_eq!(q_rev, m_rev, "reverse order mismatch at step {step}");
    assert_eq!(
        q.front().map(|(k, v)| (*k, *v)),
        m.front().copied(),
        "front mismatch at step {step}"
    );
    assert_eq!(
        q.back().map(|(k, v)| (*k, *v)),
        m.back().copied(),
        "back mismatch at step {step}"
    );
}

#[test]
fn million_op_stress_vs_oracle() {
    const STEPS: u64 = 1_000_000;
    const VERIFY_EVERY: u64 = 4096;
    const KEY_SPACE: u32 = 64;

    let mut q: LinkedQueue<K, V> = LinkedQueue::new();
    let mut m: VecDeque<(K, V)> = VecDeque::new();

    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let has = |m: &VecDeque<(K, V)>, k: K| m.iter().any(|(mk, _)| *mk == k);

    for step in 0..STEPS {
        let op = rng() % 8;
        let k = (rng() as u32) % KEY_SPACE;
        let v = rng() as u32;
        match op {
            0 => {
                let old = q.push_back(k, v);
                let model_old = model_remove(&mut m, k);
                m.push_back((k, v));
                assert_eq!(old, model_old, "push_back old @ {step}");
            }
            1 => {
                let old = q.push_front(k, v);
                let model_old = model_remove(&mut m, k);
                m.push_front((k, v));
                assert_eq!(old, model_old, "push_front old @ {step}");
            }
            2 => assert_eq!(q.pop_front(), m.pop_front(), "pop_front @ {step}"),
            3 => assert_eq!(q.pop_back(), m.pop_back(), "pop_back @ {step}"),
            4 => assert_eq!(q.remove(&k), model_remove(&mut m, k), "remove @ {step}"),
            5 => {
                let moved = q.move_to_back(&k);
                match model_remove(&mut m, k) {
                    Some(val) => {
                        assert!(moved, "move_to_back present @ {step}");
                        m.push_back((k, val));
                    }
                    None => assert!(!moved, "move_to_back absent @ {step}"),
                }
            }
            6 => {
                if let Some((anchor, _)) = m.front().copied() {
                    let r = q.insert_after(&anchor, k, v);
                    if has(&m, k) {
                        assert!(r.is_err(), "insert_after dup @ {step}");
                    } else {
                        assert!(r.is_ok(), "insert_after ok @ {step}");
                        let pos = m.iter().position(|(mk, _)| *mk == anchor).unwrap();
                        m.insert(pos + 1, (k, v));
                    }
                }
            }
            _ => {
                if let Some((anchor, _)) = m.back().copied() {
                    let r = q.insert_before(&anchor, k, v);
                    if has(&m, k) {
                        assert!(r.is_err(), "insert_before dup @ {step}");
                    } else {
                        assert!(r.is_ok(), "insert_before ok @ {step}");
                        let pos = m.iter().position(|(mk, _)| *mk == anchor).unwrap();
                        m.insert(pos, (k, v));
                    }
                }
            }
        }

        if step % VERIFY_EVERY == 0 {
            assert_equiv(&q, &m, step);
        }
    }

    assert_equiv(&q, &m, STEPS);
    // The arena must not have ballooned far beyond the live key space.
    assert!(
        q.capacity() <= (KEY_SPACE as usize) * 4,
        "arena capacity {} unexpectedly large",
        q.capacity()
    );
}

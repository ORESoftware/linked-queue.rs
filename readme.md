# linked-queue

[![CI](https://github.com/ORESoftware/linked-queue.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ORESoftware/linked-queue.rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/linked-queue.svg)](https://crates.io/crates/linked-queue)
[![docs.rs](https://docs.rs/linked-queue/badge.svg)](https://docs.rs/linked-queue)

### Constant-time queue (Rust)

A FIFO queue backed by a doubly-linked list **plus a hash index**, giving you
**O(1)** push/pop at *both* ends and — the headline feature — **O(1)** lookup,
insertion, and removal of *arbitrary* elements **by key**, from anywhere in the
queue.

This is the Rust port of the Node.js library
[`@oresoftware/linked-queue`](https://github.com/ORESoftware/linked-queue).

```bash
cargo add linked-queue
```

## Why?

Removing or inserting in the *middle* of a queue is the operation that ordinary
collections make expensive:

| Collection                          | push/pop ends | get by key | remove from middle |
| ----------------------------------- | ------------- | ---------- | ------------------ |
| `Vec` (as a queue)                  | `O(n)` shift  | —          | `O(n)`             |
| `VecDeque`                          | `O(1)`        | —          | `O(n)`             |
| `LinkedList`                        | `O(1)`        | —          | `O(n)` to *find*   |
| **`linked_queue::LinkedQueue`**     | **`O(1)`**    | **`O(1)`** | **`O(1)`**         |

The motivating use case is the [`live-mutex`](https://github.com/ORESoftware/live-mutex)
broker: pending lock requests sit in a per-key FIFO queue, and when a client
times out or disconnects, its request has to be yanked out of the middle of that
queue in constant time.

## Complexity

| Operation                              | Cost |
| -------------------------------------- | ---- |
| `push_back` / `push_front`             | O(1) |
| `pop_front` / `pop_back`               | O(1) |
| `front` / `back`                       | O(1) |
| `move_to_back` / `move_to_front`       | O(1) |
| `get` / `get_mut` / `contains`         | O(1) |
| `remove` (by key, from anywhere)       | O(1) |
| `insert_after` / `insert_before`       | O(1) |
| `iter` / `retain` / `to_vec`           | O(n) |
| `shrink_to_fit` (reclaim arena memory) | O(n) |

## Example

```rust
use linked_queue::LinkedQueue;

let mut q: LinkedQueue<&str, i32> = LinkedQueue::new();

q.push_back("a", 1);   // infallible enqueue at the tail — no `.unwrap()`
q.push_back("b", 2);
q.push_back("c", 3);

// O(1) lookup and removal from the middle, by key:
assert_eq!(q.get(&"b"), Some(&2));
assert_eq!(q.remove(&"b"), Some(2));

// O(1) splice next to an existing element:
q.insert_after(&"a", "a2", 10).unwrap();

// Still FIFO for everything else:
assert_eq!(q.pop_front(), Some(("a", 1)));
assert_eq!(q.pop_front(), Some(("a2", 10)));
assert_eq!(q.pop_front(), Some(("c", 3)));
assert_eq!(q.pop_front(), None);
```

### Insertion semantics

`push_back` / `push_front` are **infallible** — they behave like the push of any
ordinary queue and never make you handle a `Result`. Because keys are unique, a
repeated key is an *upsert*: the value is replaced and the element moves to that
end, returning the previous value. That makes `LinkedQueue` a drop-in backbone
for an **LRU cache** (`push_back` to insert/touch, `pop_front` to evict oldest):

```rust
use linked_queue::LinkedQueue;

let mut cache: LinkedQueue<&str, u32> = LinkedQueue::new();
cache.push_back("a", 1);
cache.push_back("b", 2);
assert_eq!(cache.push_back("a", 11), Some(1)); // updates "a", moves it to the back
assert_eq!(cache.pop_front(), Some(("b", 2))); // "b" is now the oldest
```

When a duplicate key should be a hard error instead of an update, use the strict
`try_push_back` / `try_push_front`, which return the rejected `(key, value)` via
`InsertError`. To update a value *without* moving the element, use `get_mut`.

## API at a glance

Every element has a **unique key** `K` and a value `V`. Keys must be
`Eq + Hash + Clone`.

- **Enqueue / push (infallible, upsert):** `push_back`, `push_front`
- **Enqueue / push (strict, rejects duplicates):** `try_push_back`,
  `try_push_front`
- **Dequeue / pop:** `pop_front` (FIFO dequeue), `pop_back`
- **Reorder, O(1):** `move_to_back`, `move_to_front` (LRU "touch")
- **By-key, O(1):** `get`, `get_mut`, `contains`, `remove`
- **O(1) splicing:** `insert_after(anchor, key, value)`,
  `insert_before(anchor, key, value)` — note: these are *implemented* here,
  unlike the upstream JS library where they currently throw "not yet
  implemented".
- **Peek:** `front`, `back`, `front_mut`, `back_mut`
- **Iterate:** `iter` (double-ended), `keys`, `values`, `into_iter`, `drain`
- **Bulk:** `retain`, `clear`, `to_vec`, `map`, `len`, `is_empty`
- **Traits:** `Default`, `Clone`, `Debug`, `PartialEq`/`Eq`, `FromIterator`,
  `Extend`, `IntoIterator`

The strict `try_push_*` and the anchored `insert_after`/`insert_before` return
`Result<(), InsertError<K, V>>`; on failure (duplicate key, or a missing anchor)
the rejected `(key, value)` is handed back to you via `InsertError`, so nothing
is ever silently dropped.

## Implementation

The queue is an **arena-backed** doubly-linked list: nodes live in a `Vec`, the
links are `usize` indices, and freed slots are recycled through a free-list so
churn doesn't keep growing the backing storage. A `HashMap<K, index>` provides
the O(1) by-key operations.

The whole crate is `#![forbid(unsafe_code)]` — no raw pointers, no
`Rc`/`RefCell` — and the linked-list invariants are checked against a
`VecDeque` oracle by a 50,000-step randomized test.

## License

MIT © ORESoftware

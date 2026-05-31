//! A FIFO queue backed by a doubly-linked list plus a hash index, giving
//! **O(1)** push/pop at *both* ends and **O(1)** lookup, insertion, and
//! removal of *arbitrary* elements by key.
//!
//! This is a Rust port of the Node.js library
//! [`@oresoftware/linked-queue`](https://github.com/ORESoftware/linked-queue).
//! Every element is associated with a unique key `K`; a [`HashMap`] maps each
//! key to a node in the list, so you can `get`, `contains`, `remove`, or
//! splice next to any element without walking the queue.
//!
//! # Why not [`std::collections::VecDeque`] or [`std::collections::LinkedList`]?
//!
//! - A `VecDeque` gives you O(1) push/pop at both ends, but removing an element
//!   from the *middle* is O(n), and there is no by-key lookup.
//! - A `LinkedList` gives you O(1) splicing only if you already hold a
//!   `Cursor`; there is no by-key lookup, and finding the element is O(n).
//!
//! `LinkedQueue` keeps a `HashMap<K, node>` alongside the list, so the common
//! "this queued item is now stale, yank it out" operation is O(1). That is
//! exactly the pattern the upstream `live-mutex` broker relies on: pending lock
//! requests live in a per-key FIFO queue, and when a client times out or
//! disconnects its request must be removed from the middle of that queue in
//! constant time.
//!
//! # Complexity
//!
//! | Operation                                   | Cost  |
//! | ------------------------------------------- | ----- |
//! | [`push_back`] / [`push_front`]              | O(1)  |
//! | [`pop_front`] / [`pop_back`]                | O(1)  |
//! | [`front`] / [`back`]                        | O(1)  |
//! | [`move_to_back`] / [`move_to_front`]        | O(1)  |
//! | [`get`] / [`get_mut`] / [`contains`]        | O(1)  |
//! | [`remove`] (by key, from anywhere)          | O(1)  |
//! | [`insert_after`] / [`insert_before`]        | O(1)  |
//! | [`iter`] / [`retain`] / [`to_vec`]          | O(n)  |
//!
//! # Insertion semantics
//!
//! [`push_back`] / [`push_front`] are **infallible** and never force you to
//! handle a `Result` (unlike the JS original, which throws on a duplicate key).
//! Keys are unique, so a repeated key is treated as an *upsert*: the value is
//! replaced and the element is moved to that end, returning the previous value.
//! This is exactly the "insert or touch" primitive an LRU cache needs. When a
//! duplicate key should instead be a hard error, reach for the strict
//! [`try_push_back`] / [`try_push_front`] variants, which return the rejected
//! `(key, value)` untouched.
//!
//! # Implementation
//!
//! Internally the queue is an **arena-backed** doubly-linked list: nodes live
//! in a `Vec<Slot>` and the `head`/`tail`/`prev`/`next` links are `usize`
//! indices rather than pointers. Freed slots are recycled through a free-list,
//! so steady-state churn does not keep growing the backing `Vec`. This keeps
//! the whole structure `#![forbid(unsafe_code)]` — no `Rc`/`RefCell`, no raw
//! pointers — while still giving the constant-time guarantees above.
//!
//! # Example
//!
//! ```
//! use linked_queue::LinkedQueue;
//!
//! let mut q: LinkedQueue<&str, i32> = LinkedQueue::new();
//!
//! q.push_back("a", 1); // infallible enqueue — no `.unwrap()` needed
//! q.push_back("b", 2);
//! q.push_back("c", 3);
//!
//! // O(1) lookup / removal from the middle, by key.
//! assert_eq!(q.get(&"b"), Some(&2));
//! assert_eq!(q.remove(&"b"), Some(2));
//!
//! // Still FIFO for everything that's left.
//! assert_eq!(q.pop_front(), Some(("a", 1)));
//! assert_eq!(q.pop_front(), Some(("c", 3)));
//! assert_eq!(q.pop_front(), None);
//! ```
//!
//! [`HashMap`]: std::collections::HashMap
//! [`push_back`]: LinkedQueue::push_back
//! [`push_front`]: LinkedQueue::push_front
//! [`try_push_back`]: LinkedQueue::try_push_back
//! [`try_push_front`]: LinkedQueue::try_push_front
//! [`move_to_back`]: LinkedQueue::move_to_back
//! [`move_to_front`]: LinkedQueue::move_to_front
//! [`pop_front`]: LinkedQueue::pop_front
//! [`pop_back`]: LinkedQueue::pop_back
//! [`front`]: LinkedQueue::front
//! [`back`]: LinkedQueue::back
//! [`get`]: LinkedQueue::get
//! [`get_mut`]: LinkedQueue::get_mut
//! [`contains`]: LinkedQueue::contains
//! [`remove`]: LinkedQueue::remove
//! [`insert_after`]: LinkedQueue::insert_after
//! [`insert_before`]: LinkedQueue::insert_before
//! [`iter`]: LinkedQueue::iter
//! [`retain`]: LinkedQueue::retain
//! [`to_vec`]: LinkedQueue::to_vec

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Sentinel index meaning "no node" (the linked-list equivalent of a null
/// pointer). `usize::MAX` is safe to use because a real slot index can never
/// reach it: that would require `usize::MAX` live slots, which cannot be
/// allocated.
const NIL: usize = usize::MAX;

/// One node in the arena. `key`/`value` are `Option` so a freed slot (sitting
/// on the free-list) can hold `None` without dropping into `unsafe`.
struct Slot<K, V> {
    key: Option<K>,
    value: Option<V>,
    prev: usize,
    next: usize,
}

/// Error returned by the fallible insertion methods when an element could not
/// be inserted. The rejected `key` and `value` are handed back so the caller
/// never loses ownership of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError<K, V> {
    /// The `key` is already present in the queue. Each key must be unique;
    /// remove the existing element first if you want to re-insert.
    DuplicateKey {
        /// The key that was rejected.
        key: K,
        /// The value that was rejected.
        value: V,
    },
    /// The anchor key passed to [`insert_after`](LinkedQueue::insert_after) or
    /// [`insert_before`](LinkedQueue::insert_before) is not present in the
    /// queue, so there was nothing to splice next to.
    AnchorMissing {
        /// The key that was rejected.
        key: K,
        /// The value that was rejected.
        value: V,
    },
}

impl<K, V> InsertError<K, V> {
    /// Recover the rejected `(key, value)` pair, discarding the error kind.
    pub fn into_inner(self) -> (K, V) {
        match self {
            InsertError::DuplicateKey { key, value } => (key, value),
            InsertError::AnchorMissing { key, value } => (key, value),
        }
    }
}

impl<K, V> fmt::Display for InsertError<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InsertError::DuplicateKey { .. } => f.write_str("key is already present in the queue"),
            InsertError::AnchorMissing { .. } => {
                f.write_str("anchor key is not present in the queue")
            }
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> std::error::Error for InsertError<K, V> {}

/// A FIFO queue keyed by `K`, backed by a doubly-linked list with an O(1)
/// hash index.
///
/// See the [crate-level docs](crate) for the design rationale and a complexity
/// table. Keys are unique: [`push_back`](Self::push_back) /
/// [`push_front`](Self::push_front) upsert a repeated key (replacing its value
/// and moving the element), while the [`try_push_back`](Self::try_push_back) /
/// [`try_push_front`](Self::try_push_front) variants reject it.
pub struct LinkedQueue<K, V> {
    slots: Vec<Slot<K, V>>,
    free: Vec<usize>,
    index: HashMap<K, usize>,
    head: usize,
    tail: usize,
    len: usize,
}

impl<K: Eq + Hash + Clone, V> LinkedQueue<K, V> {
    /// Create an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            index: HashMap::new(),
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    /// Create an empty queue that can hold at least `capacity` elements without
    /// reallocating its arena or index.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            index: HashMap::with_capacity(capacity),
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    /// Number of elements currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the queue holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of elements the arena can hold before it must reallocate.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Reserve space for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
        self.index.reserve(additional);
    }

    /// `true` if `key` is present anywhere in the queue. O(1).
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Borrow the value associated with `key`, if present. O(1).
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        let idx = *self.index.get(key)?;
        self.slots[idx].value.as_ref()
    }

    /// Mutably borrow the value associated with `key`, if present. O(1).
    ///
    /// The element keeps its position in the queue.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let idx = *self.index.get(key)?;
        self.slots[idx].value.as_mut()
    }

    /// Borrow the head (front) element without removing it. O(1).
    #[must_use]
    pub fn front(&self) -> Option<(&K, &V)> {
        self.borrow_slot(self.head)
    }

    /// Borrow the tail (back) element without removing it. O(1).
    #[must_use]
    pub fn back(&self) -> Option<(&K, &V)> {
        self.borrow_slot(self.tail)
    }

    /// Mutably borrow the head element's value (with read access to its key)
    /// without removing it or changing its position. O(1).
    pub fn front_mut(&mut self) -> Option<(&K, &mut V)> {
        Self::borrow_slot_mut(&mut self.slots, self.head)
    }

    /// Mutably borrow the tail element's value (with read access to its key)
    /// without removing it or changing its position. O(1).
    pub fn back_mut(&mut self) -> Option<(&K, &mut V)> {
        Self::borrow_slot_mut(&mut self.slots, self.tail)
    }

    /// Enqueue at the tail. O(1). Infallible — this is the everyday "add to the
    /// queue" operation and never forces error handling.
    ///
    /// - If `key` is **new**, the element is appended at the tail and `None` is
    ///   returned.
    /// - If `key` is **already present**, its value is replaced *and* the
    ///   element is moved to the tail; the previous value is returned as
    ///   `Some(old)`. This "insert-or-touch" behaviour is what makes the type a
    ///   ready-made backbone for an LRU cache (`push_back` to insert/touch,
    ///   [`pop_front`](Self::pop_front) to evict the oldest).
    ///
    /// If a duplicate key should instead be *rejected*, use
    /// [`try_push_back`](Self::try_push_back). To update a value **without**
    /// moving the element, use [`get_mut`](Self::get_mut).
    ///
    /// ```
    /// # use linked_queue::LinkedQueue;
    /// let mut q = LinkedQueue::new();
    /// assert_eq!(q.push_back("a", 1), None);
    /// assert_eq!(q.push_back("b", 2), None);
    /// assert_eq!(q.push_back("a", 9), Some(1)); // updates value, moves "a" to the back
    /// assert_eq!(q.to_vec(), vec![("b", 2), ("a", 9)]);
    /// ```
    pub fn push_back(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&idx) = self.index.get(&key) {
            let old = self.slots[idx].value.replace(value);
            self.unlink(idx);
            self.link_back(idx);
            return old;
        }
        let idx = self.alloc(key.clone(), value);
        self.link_back(idx);
        self.index.insert(key, idx);
        self.len += 1;
        None
    }

    /// Enqueue at the head (jump the queue). O(1). Infallible.
    ///
    /// Mirrors [`push_back`](Self::push_back): a new key is prepended; an
    /// existing key has its value replaced and is moved to the head, returning
    /// the previous value as `Some(old)`.
    pub fn push_front(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&idx) = self.index.get(&key) {
            let old = self.slots[idx].value.replace(value);
            self.unlink(idx);
            self.link_front(idx);
            return old;
        }
        let idx = self.alloc(key.clone(), value);
        self.link_front(idx);
        self.index.insert(key, idx);
        self.len += 1;
        None
    }

    /// Strict enqueue at the tail. O(1).
    ///
    /// Behaves like [`push_back`](Self::push_back) for a new key, but if `key`
    /// is already present the queue is left **unchanged** and the rejected
    /// `(key, value)` is handed back via [`InsertError::DuplicateKey`]. Use this
    /// when keys are an invariant (e.g. unique request IDs) and a collision is a
    /// bug, not an update.
    ///
    /// ```
    /// # use linked_queue::{LinkedQueue, InsertError};
    /// let mut q = LinkedQueue::new();
    /// q.try_push_back("a", 1).unwrap();
    /// assert_eq!(
    ///     q.try_push_back("a", 2),
    ///     Err(InsertError::DuplicateKey { key: "a", value: 2 }),
    /// );
    /// assert_eq!(q.get(&"a"), Some(&1));
    /// ```
    pub fn try_push_back(&mut self, key: K, value: V) -> Result<(), InsertError<K, V>> {
        if self.index.contains_key(&key) {
            return Err(InsertError::DuplicateKey { key, value });
        }
        let idx = self.alloc(key.clone(), value);
        self.link_back(idx);
        self.index.insert(key, idx);
        self.len += 1;
        Ok(())
    }

    /// Strict enqueue at the head. O(1). The head-side counterpart of
    /// [`try_push_back`](Self::try_push_back).
    pub fn try_push_front(&mut self, key: K, value: V) -> Result<(), InsertError<K, V>> {
        if self.index.contains_key(&key) {
            return Err(InsertError::DuplicateKey { key, value });
        }
        let idx = self.alloc(key.clone(), value);
        self.link_front(idx);
        self.index.insert(key, idx);
        self.len += 1;
        Ok(())
    }

    /// Move an existing element to the tail without touching its value. O(1).
    /// Returns `true` if `key` was present. This is the "mark most-recently
    /// used" operation for LRU-style usage.
    pub fn move_to_back(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.index.get(key) {
            self.unlink(idx);
            self.link_back(idx);
            true
        } else {
            false
        }
    }

    /// Move an existing element to the head without touching its value. O(1).
    /// Returns `true` if `key` was present.
    pub fn move_to_front(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.index.get(key) {
            self.unlink(idx);
            self.link_front(idx);
            true
        } else {
            false
        }
    }

    /// Insert `(key, value)` immediately **after** the element keyed by
    /// `anchor`. O(1).
    ///
    /// Unlike the upstream JS library (where `insertInFrontOf` is unimplemented
    /// and throws), this is a real constant-time splice.
    ///
    /// # Errors
    /// - [`InsertError::DuplicateKey`] if `key` already exists.
    /// - [`InsertError::AnchorMissing`] if `anchor` is not in the queue.
    ///
    /// ```
    /// # use linked_queue::LinkedQueue;
    /// let mut q = LinkedQueue::new();
    /// q.push_back("a", 1);
    /// q.push_back("c", 3);
    /// q.insert_after(&"a", "b", 2).unwrap();
    /// assert_eq!(q.to_vec(), vec![("a", 1), ("b", 2), ("c", 3)]);
    /// ```
    pub fn insert_after(&mut self, anchor: &K, key: K, value: V) -> Result<(), InsertError<K, V>> {
        if self.index.contains_key(&key) {
            return Err(InsertError::DuplicateKey { key, value });
        }
        let Some(&a) = self.index.get(anchor) else {
            return Err(InsertError::AnchorMissing { key, value });
        };
        let idx = self.alloc(key.clone(), value);
        let after = self.slots[a].next;
        self.slots[idx].prev = a;
        self.slots[idx].next = after;
        self.slots[a].next = idx;
        if after == NIL {
            self.tail = idx;
        } else {
            self.slots[after].prev = idx;
        }
        self.index.insert(key, idx);
        self.len += 1;
        Ok(())
    }

    /// Insert `(key, value)` immediately **before** the element keyed by
    /// `anchor`. O(1).
    ///
    /// # Errors
    /// - [`InsertError::DuplicateKey`] if `key` already exists.
    /// - [`InsertError::AnchorMissing`] if `anchor` is not in the queue.
    pub fn insert_before(&mut self, anchor: &K, key: K, value: V) -> Result<(), InsertError<K, V>> {
        if self.index.contains_key(&key) {
            return Err(InsertError::DuplicateKey { key, value });
        }
        let Some(&a) = self.index.get(anchor) else {
            return Err(InsertError::AnchorMissing { key, value });
        };
        let idx = self.alloc(key.clone(), value);
        let before = self.slots[a].prev;
        self.slots[idx].next = a;
        self.slots[idx].prev = before;
        self.slots[a].prev = idx;
        if before == NIL {
            self.head = idx;
        } else {
            self.slots[before].next = idx;
        }
        self.index.insert(key, idx);
        self.len += 1;
        Ok(())
    }

    /// Remove and return the head element (standard FIFO dequeue). O(1).
    pub fn pop_front(&mut self) -> Option<(K, V)> {
        if self.head == NIL {
            return None;
        }
        Some(self.remove_at(self.head))
    }

    /// Remove and return the tail element. O(1).
    pub fn pop_back(&mut self) -> Option<(K, V)> {
        if self.tail == NIL {
            return None;
        }
        Some(self.remove_at(self.tail))
    }

    /// Remove the element keyed by `key` from anywhere in the queue, returning
    /// its value. O(1). Returns `None` if the key is absent.
    ///
    /// ```
    /// # use linked_queue::LinkedQueue;
    /// let mut q = LinkedQueue::new();
    /// for k in ["a", "b", "c", "d"] { q.push_back(k, k.to_string()); }
    /// assert_eq!(q.remove(&"c"), Some("c".to_string()));
    /// assert_eq!(q.remove(&"z"), None);
    /// assert_eq!(q.len(), 3);
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = *self.index.get(key)?;
        Some(self.remove_at(idx).1)
    }

    /// Remove every element, keeping the queue allocated for reuse.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.index.clear();
        self.head = NIL;
        self.tail = NIL;
        self.len = 0;
    }

    /// Retain only the elements for which `f` returns `true`, in queue order.
    /// O(n). Removed elements are dropped.
    pub fn retain<F: FnMut(&K, &V) -> bool>(&mut self, mut f: F) {
        let mut cur = self.head;
        while cur != NIL {
            let next = self.slots[cur].next;
            let keep = {
                let s = &self.slots[cur];
                f(
                    s.key.as_ref().expect("occupied slot"),
                    s.value.as_ref().expect("occupied slot"),
                )
            };
            if !keep {
                self.remove_at(cur);
            }
            cur = next;
        }
    }

    /// Iterate over `(&K, &V)` from head to tail without consuming the queue.
    /// Implements [`DoubleEndedIterator`], so `.rev()` walks tail to head.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            queue: self,
            front: self.head,
            back: self.tail,
            remaining: self.len,
        }
    }

    /// Iterate over the keys from head to tail.
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over the values from head to tail.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Drain the queue from the head, yielding owned `(K, V)` pairs. Whatever
    /// is not consumed is dropped when the [`Drain`] iterator is dropped, so
    /// the queue is always empty afterwards.
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        Drain { queue: self }
    }

    /// Collect the elements into a `Vec<(K, V)>` in head-to-tail order. O(n).
    #[must_use]
    pub fn to_vec(&self) -> Vec<(K, V)>
    where
        V: Clone,
    {
        self.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Map each element to `R` in head-to-tail order, collecting into a `Vec`.
    pub fn map<R, F: FnMut(&K, &V) -> R>(&self, mut f: F) -> Vec<R> {
        self.iter().map(|(k, v)| f(k, v)).collect()
    }

    // ---- private helpers --------------------------------------------------

    fn borrow_slot(&self, idx: usize) -> Option<(&K, &V)> {
        if idx == NIL {
            return None;
        }
        let s = &self.slots[idx];
        Some((s.key.as_ref()?, s.value.as_ref()?))
    }

    /// Disjoint split-borrow of one slot's key (`&K`) and value (`&mut V`).
    /// Taken as an associated fn over `&mut [Slot]` so it borrows only the
    /// arena, leaving the rest of `self` free.
    fn borrow_slot_mut(slots: &mut [Slot<K, V>], idx: usize) -> Option<(&K, &mut V)> {
        if idx == NIL {
            return None;
        }
        let s = &mut slots[idx];
        match (&s.key, &mut s.value) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Grab a slot for a new node, reusing a freed one when possible.
    fn alloc(&mut self, key: K, value: V) -> usize {
        if let Some(idx) = self.free.pop() {
            let slot = &mut self.slots[idx];
            slot.key = Some(key);
            slot.value = Some(value);
            slot.prev = NIL;
            slot.next = NIL;
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Slot {
                key: Some(key),
                value: Some(value),
                prev: NIL,
                next: NIL,
            });
            idx
        }
    }

    /// Detach the node at `idx` from the list, repairing the neighbour and
    /// head/tail links. The node's own `prev`/`next` are left dangling for the
    /// caller to either reset (via a `link_*` call) or recycle (`remove_at`).
    fn unlink(&mut self, idx: usize) {
        let prev = self.slots[idx].prev;
        let next = self.slots[idx].next;
        if prev == NIL {
            self.head = next;
        } else {
            self.slots[prev].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.slots[next].prev = prev;
        }
    }

    /// Link an already-allocated, currently-detached node at the tail.
    fn link_back(&mut self, idx: usize) {
        self.slots[idx].next = NIL;
        if self.tail == NIL {
            self.slots[idx].prev = NIL;
            self.head = idx;
        } else {
            let t = self.tail;
            self.slots[t].next = idx;
            self.slots[idx].prev = t;
        }
        self.tail = idx;
    }

    /// Link an already-allocated, currently-detached node at the head.
    fn link_front(&mut self, idx: usize) {
        self.slots[idx].prev = NIL;
        if self.head == NIL {
            self.slots[idx].next = NIL;
            self.tail = idx;
        } else {
            let h = self.head;
            self.slots[h].prev = idx;
            self.slots[idx].next = h;
        }
        self.head = idx;
    }

    /// Unlink the node at `idx`, return its `(K, V)`, and recycle the slot.
    fn remove_at(&mut self, idx: usize) -> (K, V) {
        self.unlink(idx);
        let key = self.slots[idx].key.take().expect("occupied slot has a key");
        let value = self.slots[idx]
            .value
            .take()
            .expect("occupied slot has a value");
        self.slots[idx].prev = NIL;
        self.slots[idx].next = NIL;
        self.index.remove(&key);
        self.free.push(idx);
        self.len -= 1;
        (key, value)
    }
}

// ---- iterators ------------------------------------------------------------

/// Borrowing iterator over `(&K, &V)`, head to tail. Created by
/// [`LinkedQueue::iter`].
pub struct Iter<'a, K, V> {
    queue: &'a LinkedQueue<K, V>,
    front: usize,
    back: usize,
    remaining: usize,
}

impl<'a, K, V> fmt::Debug for Iter<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter")
            .field("remaining", &self.remaining)
            .finish()
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let slot = &self.queue.slots[self.front];
        self.front = slot.next;
        self.remaining -= 1;
        Some((
            slot.key.as_ref().expect("occupied slot"),
            slot.value.as_ref().expect("occupied slot"),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let slot = &self.queue.slots[self.back];
        self.back = slot.prev;
        self.remaining -= 1;
        Some((
            slot.key.as_ref().expect("occupied slot"),
            slot.value.as_ref().expect("occupied slot"),
        ))
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}
impl<'a, K, V> std::iter::FusedIterator for Iter<'a, K, V> {}

/// Owning iterator over `(K, V)`, head to tail. Created by
/// [`IntoIterator::into_iter`] on an owned [`LinkedQueue`].
pub struct IntoIter<K, V> {
    queue: LinkedQueue<K, V>,
}

impl<K: Eq + Hash + Clone, V> fmt::Debug for IntoIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("remaining", &self.queue.len())
            .finish()
    }
}

impl<K: Eq + Hash + Clone, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.queue.len(), Some(self.queue.len()))
    }
}

impl<K: Eq + Hash + Clone, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.queue.pop_back()
    }
}

impl<K: Eq + Hash + Clone, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K: Eq + Hash + Clone, V> std::iter::FusedIterator for IntoIter<K, V> {}

/// Draining iterator created by [`LinkedQueue::drain`]. Pops from the head;
/// on drop it clears whatever remains.
pub struct Drain<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    queue: &'a mut LinkedQueue<K, V>,
}

impl<'a, K: Eq + Hash + Clone, V> fmt::Debug for Drain<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Drain")
            .field("remaining", &self.queue.len())
            .finish()
    }
}

impl<'a, K: Eq + Hash + Clone, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.queue.len(), Some(self.queue.len()))
    }
}

impl<'a, K: Eq + Hash + Clone, V> DoubleEndedIterator for Drain<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.queue.pop_back()
    }
}

impl<'a, K: Eq + Hash + Clone, V> ExactSizeIterator for Drain<'a, K, V> {}
impl<'a, K: Eq + Hash + Clone, V> std::iter::FusedIterator for Drain<'a, K, V> {}

impl<'a, K: Eq + Hash + Clone, V> Drop for Drain<'a, K, V> {
    fn drop(&mut self) {
        self.queue.clear();
    }
}

// ---- standard trait impls -------------------------------------------------

impl<K: Eq + Hash + Clone, V> Default for LinkedQueue<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> IntoIterator for LinkedQueue<K, V>
where
    K: Eq + Hash + Clone,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { queue: self }
    }
}

impl<'a, K, V> IntoIterator for &'a LinkedQueue<K, V>
where
    K: Eq + Hash + Clone,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Clone for LinkedQueue<K, V> {
    fn clone(&self) -> Self {
        let mut q = LinkedQueue::with_capacity(self.len);
        for (k, v) in self.iter() {
            // Source keys are already unique, so every push is a fresh insert
            // at the tail and the returned `None` is expected.
            q.push_back(k.clone(), v.clone());
        }
        q
    }
}

impl<K, V> fmt::Debug for LinkedQueue<K, V>
where
    K: Eq + Hash + Clone + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> PartialEq for LinkedQueue<K, V>
where
    K: Eq + Hash + Clone,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<K, V> Eq for LinkedQueue<K, V>
where
    K: Eq + Hash + Clone,
    V: Eq,
{
}

impl<K: Eq + Hash + Clone, V> FromIterator<(K, V)> for LinkedQueue<K, V> {
    /// Build a queue from `(key, value)` pairs in iteration order, via
    /// [`push_back`](LinkedQueue::push_back). If the same key appears more than
    /// once, the **last** value wins and that entry ends up at the back
    /// (consistent with `push_back`'s upsert semantics).
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut q = LinkedQueue::new();
        q.extend(iter);
        q
    }
}

impl<K: Eq + Hash + Clone, V> Extend<(K, V)> for LinkedQueue<K, V> {
    /// Push each `(key, value)` at the tail via
    /// [`push_back`](LinkedQueue::push_back). A repeated key updates the value
    /// and moves that entry to the back (last value wins).
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let it = iter.into_iter();
        self.reserve(it.size_hint().0);
        for (k, v) in it {
            self.push_back(k, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn fifo_order() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        assert_eq!(q.push_back("a", 1), None);
        assert_eq!(q.push_back("b", 2), None);
        assert_eq!(q.push_back("c", 3), None);
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop_front(), Some(("a", 1)));
        assert_eq!(q.pop_front(), Some(("b", 2)));
        assert_eq!(q.pop_front(), Some(("c", 3)));
        assert_eq!(q.pop_front(), None);
        assert!(q.is_empty());
    }

    #[test]
    fn push_back_upserts_and_moves_to_tail() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        q.push_back("c", 3);
        // Re-pushing an existing key updates the value, returns the old one,
        // and moves the element to the back.
        assert_eq!(q.push_back("a", 10), Some(1));
        assert_eq!(q.to_vec(), vec![("b", 2), ("c", 3), ("a", 10)]);
        assert_eq!(q.len(), 3);

        // push_front mirrors that toward the head.
        assert_eq!(q.push_front("c", 30), Some(3));
        assert_eq!(q.to_vec(), vec![("c", 30), ("b", 2), ("a", 10)]);
    }

    #[test]
    fn try_push_rejects_duplicate_and_returns_value() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.try_push_back("a", 1).unwrap();
        let err = q.try_push_back("a", 99).unwrap_err();
        assert_eq!(
            err,
            InsertError::DuplicateKey {
                key: "a",
                value: 99
            }
        );
        assert_eq!(err.into_inner(), ("a", 99));
        // try_push leaves the existing element and order untouched.
        assert_eq!(q.try_push_front("a", 7).unwrap_err().into_inner(), ("a", 7));
        assert_eq!(q.len(), 1);
        assert_eq!(q.get(&"a"), Some(&1));
    }

    #[test]
    fn lru_eviction_pattern() {
        // push_back = insert/touch, pop_front = evict oldest.
        let mut cache: LinkedQueue<&str, u32> = LinkedQueue::new();
        cache.push_back("a", 1);
        cache.push_back("b", 2);
        cache.push_back("c", 3);
        cache.push_back("a", 11); // touch "a" -> now most-recently-used
                                  // Oldest is "b" now.
        assert_eq!(cache.pop_front(), Some(("b", 2)));
        assert_eq!(cache.to_vec(), vec![("c", 3), ("a", 11)]);
    }

    #[test]
    fn remove_from_middle_is_o1_correct() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        q.push_back("c", 3);
        q.push_back("d", 4);
        assert_eq!(q.remove(&"b"), Some(2));
        assert_eq!(q.remove(&"d"), Some(4));
        assert_eq!(q.remove(&"z"), None);
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop_front(), Some(("a", 1)));
        assert_eq!(q.pop_front(), Some(("c", 3)));
        assert_eq!(q.pop_front(), None);
    }

    #[test]
    fn remove_head_and_tail() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        q.push_back("c", 3);
        assert_eq!(q.remove(&"a"), Some(1));
        assert_eq!(q.remove(&"c"), Some(3));
        assert_eq!(q.len(), 1);
        assert_eq!(q.front(), Some((&"b", &2)));
        assert_eq!(q.back(), Some((&"b", &2)));
        assert_eq!(q.pop_back(), Some(("b", 2)));
    }

    #[test]
    fn push_front_and_pop_back() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        q.push_front("c", 3);
        assert_eq!(q.to_vec(), vec![("c", 3), ("a", 1), ("b", 2)]);
        assert_eq!(q.pop_back(), Some(("b", 2)));
        assert_eq!(q.pop_front(), Some(("c", 3)));
        assert_eq!(q.pop_front(), Some(("a", 1)));
    }

    #[test]
    fn insert_after_and_before() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("d", 4);
        q.insert_after(&"a", "b", 2).unwrap();
        q.insert_before(&"d", "c", 3).unwrap();
        assert_eq!(q.to_vec(), vec![("a", 1), ("b", 2), ("c", 3), ("d", 4)]);

        // insert at the very front / back via anchors on the ends
        q.insert_before(&"a", "z", 0).unwrap();
        q.insert_after(&"d", "e", 5).unwrap();
        assert_eq!(q.front(), Some((&"z", &0)));
        assert_eq!(q.back(), Some((&"e", &5)));
    }

    #[test]
    fn insert_errors() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        assert_eq!(
            q.insert_after(&"a", "a", 9).unwrap_err(),
            InsertError::DuplicateKey { key: "a", value: 9 }
        );
        assert_eq!(
            q.insert_after(&"missing", "b", 2).unwrap_err(),
            InsertError::AnchorMissing { key: "b", value: 2 }
        );
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn get_mut_updates_in_place() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        *q.get_mut(&"a").unwrap() += 100;
        assert_eq!(q.get(&"a"), Some(&101));
        // order preserved (unlike push_back, get_mut never moves the element)
        assert_eq!(q.to_vec(), vec![("a", 101), ("b", 2)]);
    }

    #[test]
    fn front_back_mut() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        assert!(q.front_mut().is_none());
        q.push_back("a", 1);
        q.push_back("b", 2);
        if let Some((k, v)) = q.front_mut() {
            assert_eq!(*k, "a");
            *v += 100;
        }
        if let Some((k, v)) = q.back_mut() {
            assert_eq!(*k, "b");
            *v += 200;
        }
        assert_eq!(q.to_vec(), vec![("a", 101), ("b", 202)]);
    }

    #[test]
    fn move_to_back_and_front() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        q.push_back("c", 3);
        assert!(q.move_to_back(&"a"));
        assert_eq!(q.to_vec(), vec![("b", 2), ("c", 3), ("a", 1)]);
        assert!(q.move_to_front(&"c"));
        assert_eq!(q.to_vec(), vec![("c", 3), ("b", 2), ("a", 1)]);
        // unknown key is a no-op
        assert!(!q.move_to_back(&"z"));
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn double_ended_iteration() {
        let mut q: LinkedQueue<u32, u32> = LinkedQueue::new();
        for i in 0..5 {
            q.push_back(i, i * 10);
        }
        let fwd: Vec<_> = q.iter().map(|(k, v)| (*k, *v)).collect();
        let rev: Vec<_> = q.iter().rev().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(fwd, vec![(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]);
        assert_eq!(rev, vec![(4, 40), (3, 30), (2, 20), (1, 10), (0, 0)]);

        // meet in the middle
        let mut it = q.iter();
        assert_eq!(it.next().map(|(k, _)| *k), Some(0));
        assert_eq!(it.next_back().map(|(k, _)| *k), Some(4));
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn retain_keeps_order() {
        let mut q: LinkedQueue<u32, u32> = LinkedQueue::new();
        for i in 0..10 {
            q.push_back(i, i);
        }
        q.retain(|_, v| v % 2 == 0);
        assert_eq!(q.to_vec(), vec![(0, 0), (2, 2), (4, 4), (6, 6), (8, 8)]);
        assert_eq!(q.len(), 5);
    }

    #[test]
    fn drain_empties_queue() {
        let mut q: LinkedQueue<u32, u32> = LinkedQueue::new();
        for i in 0..5 {
            q.push_back(i, i);
        }
        let first_two: Vec<_> = q.drain().take(2).collect();
        assert_eq!(first_two, vec![(0, 0), (1, 1)]);
        // drain drops the rest on drop
        assert!(q.is_empty());
    }

    #[test]
    fn into_iter_owned() {
        let mut q: LinkedQueue<u32, String> = LinkedQueue::new();
        q.push_back(1, "one".into());
        q.push_back(2, "two".into());
        let collected: Vec<_> = q.into_iter().collect();
        assert_eq!(
            collected,
            vec![(1, "one".to_string()), (2, "two".to_string())]
        );
    }

    #[test]
    fn from_iter_last_value_wins() {
        // push_back upsert semantics: the last value for "a" wins and that
        // entry ends up at the back.
        let q: LinkedQueue<&str, u32> = [("a", 1), ("b", 2), ("a", 99)].into_iter().collect();
        assert_eq!(q.len(), 2);
        assert_eq!(q.get(&"a"), Some(&99));
        assert_eq!(q.to_vec(), vec![("b", 2), ("a", 99)]);
    }

    #[test]
    fn clone_and_eq() {
        let mut q: LinkedQueue<&str, u32> = LinkedQueue::new();
        q.push_back("a", 1);
        q.push_back("b", 2);
        let c = q.clone();
        assert_eq!(q, c);
        let mut d = c.clone();
        d.push_back("c", 3);
        assert_ne!(q, d);
    }

    #[test]
    fn slots_get_reused() {
        let mut q: LinkedQueue<u32, u32> = LinkedQueue::new();
        for i in 0..1000 {
            q.push_back(i, i);
        }
        for i in 0..1000 {
            assert_eq!(q.remove(&i), Some(i));
        }
        for i in 0..1000 {
            q.push_back(i, i * 2);
        }
        // Free-list keeps total slots bounded near peak occupancy.
        assert!(
            q.capacity() <= 1024,
            "capacity grew unexpectedly: {}",
            q.capacity()
        );
        assert_eq!(q.len(), 1000);
    }

    /// Remove the entry for `k` from the model, returning its value if present.
    fn model_remove(model: &mut VecDeque<(u32, u32)>, k: u32) -> Option<u32> {
        model
            .iter()
            .position(|(mk, _)| *mk == k)
            .map(|p| model.remove(p).unwrap().1)
    }

    /// Randomised op sequence checked against a `VecDeque` oracle. Verifies
    /// that `head`/`tail`/`prev`/`next` never drift from the index: after every
    /// step the model and the queue agree on `front`, `back`, `len`,
    /// `contains`, the upsert return values, and the full iteration order (both
    /// directions). Exercises the upsert/move semantics of `push_*` plus
    /// `move_to_*`, `remove`, and the anchored inserts.
    #[test]
    fn fuzz_against_vecdeque_oracle() {
        let mut q: LinkedQueue<u32, u32> = LinkedQueue::new();
        let mut model: VecDeque<(u32, u32)> = VecDeque::new();

        // Deterministic LCG so the test is stable across toolchains.
        let mut state: u64 = 0xdead_beef_cafe_f00d;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        let has = |model: &VecDeque<(u32, u32)>, k: u32| model.iter().any(|(mk, _)| *mk == k);

        for step in 0..50_000u32 {
            let op = rng() % 9;
            let k = (rng() % 48) as u32;
            let v = (rng() % 1024) as u32;
            match op {
                0 => {
                    // push_back: upsert + move to tail, returns old value.
                    let old = q.push_back(k, v);
                    let model_old = model_remove(&mut model, k);
                    model.push_back((k, v));
                    assert_eq!(old, model_old, "step {step}: push_back old value");
                }
                1 => {
                    // push_front: upsert + move to head, returns old value.
                    let old = q.push_front(k, v);
                    let model_old = model_remove(&mut model, k);
                    model.push_front((k, v));
                    assert_eq!(old, model_old, "step {step}: push_front old value");
                }
                2 => {
                    assert_eq!(q.pop_front(), model.pop_front(), "step {step}: pop_front");
                }
                3 => {
                    assert_eq!(q.pop_back(), model.pop_back(), "step {step}: pop_back");
                }
                4 => {
                    let removed = q.remove(&k);
                    let model_removed = model_remove(&mut model, k);
                    assert_eq!(removed, model_removed, "step {step}: remove({k})");
                }
                5 => {
                    // insert_after: anchor = a random existing key (if any)
                    if let Some((anchor, _)) = model.front().copied() {
                        let r = q.insert_after(&anchor, k, v);
                        if has(&model, k) {
                            assert!(r.is_err(), "step {step}: insert_after dup");
                        } else {
                            assert!(r.is_ok(), "step {step}: insert_after ok");
                            let pos = model.iter().position(|(mk, _)| *mk == anchor).unwrap();
                            model.insert(pos + 1, (k, v));
                        }
                    }
                }
                6 => {
                    if let Some((anchor, _)) = model.back().copied() {
                        let r = q.insert_before(&anchor, k, v);
                        if has(&model, k) {
                            assert!(r.is_err(), "step {step}: insert_before dup");
                        } else {
                            assert!(r.is_ok(), "step {step}: insert_before ok");
                            let pos = model.iter().position(|(mk, _)| *mk == anchor).unwrap();
                            model.insert(pos, (k, v));
                        }
                    }
                }
                7 => {
                    // move_to_back / move_to_front, value-preserving.
                    if v % 2 == 0 {
                        let moved = q.move_to_back(&k);
                        match model_remove(&mut model, k) {
                            Some(val) => {
                                assert!(moved, "step {step}: move_to_back present");
                                model.push_back((k, val));
                            }
                            None => assert!(!moved, "step {step}: move_to_back absent"),
                        }
                    } else {
                        let moved = q.move_to_front(&k);
                        match model_remove(&mut model, k) {
                            Some(val) => {
                                assert!(moved, "step {step}: move_to_front present");
                                model.push_front((k, val));
                            }
                            None => assert!(!moved, "step {step}: move_to_front absent"),
                        }
                    }
                }
                _ => {
                    assert_eq!(q.len(), model.len(), "step {step}: len");
                    assert_eq!(q.contains(&k), has(&model, k), "step {step}: contains({k})");
                    let fwd: Vec<(u32, u32)> = q.iter().map(|(k, v)| (*k, *v)).collect();
                    let m_fwd: Vec<(u32, u32)> = model.iter().copied().collect();
                    assert_eq!(fwd, m_fwd, "step {step}: iter order");

                    let rev: Vec<(u32, u32)> = q.iter().rev().map(|(k, v)| (*k, *v)).collect();
                    let mut m_rev = m_fwd.clone();
                    m_rev.reverse();
                    assert_eq!(rev, m_rev, "step {step}: rev iter order");

                    assert_eq!(
                        q.front().map(|(k, v)| (*k, *v)),
                        model.front().copied(),
                        "step {step}: front"
                    );
                    assert_eq!(
                        q.back().map(|(k, v)| (*k, *v)),
                        model.back().copied(),
                        "step {step}: back"
                    );
                }
            }
        }
    }
}

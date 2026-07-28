//! Block state packing: block id + property bits into one `u32`, plus a
//! compact name <-> id registry.
//!
//! A full name and property map per voxel would be enormous, so every
//! block state is one `u32`, with id-to-name kept once per world.

use std::collections::HashMap;

/// Bits of a packed state reserved for the block id.
pub const ID_BITS: u32 = 12;
/// Bits of a packed state reserved for properties.
pub const PROP_BITS: u32 = 32 - ID_BITS;
/// Largest block id representable in [`ID_BITS`] bits.
pub const MAX_BLOCK_ID: u16 = ((1u32 << ID_BITS) - 1) as u16;
/// Largest property bag representable in [`PROP_BITS`] bits.
pub const MAX_PROPS: u32 = (1u32 << PROP_BITS) - 1;

/// Pack a block id and a property bag into one `u32` state: id in the low
/// [`ID_BITS`] bits, properties in the rest. Panics if either operand
/// overflows its field.
#[inline]
pub fn pack_state(id: u16, props: u32) -> u32 {
    assert!(id <= MAX_BLOCK_ID, "block id overflows its bit field");
    assert!(props <= MAX_PROPS, "properties overflow their bit field");
    (id as u32) | (props << ID_BITS)
}

/// Extract the block id from a packed state.
#[inline]
pub fn state_id(state: u32) -> u16 {
    (state & MAX_BLOCK_ID as u32) as u16
}

/// Extract the property bag from a packed state.
#[inline]
pub fn state_props(state: u32) -> u32 {
    state >> ID_BITS
}

/// View a packed state as 4 raw bytes, e.g. for writing into a region file
/// section's block-state array without an explicit `to_le_bytes` call site
/// at every call.
#[inline]
pub fn state_to_bytes(state: u32) -> [u8; 4] {
    state.to_ne_bytes()
}

/// Inverse of [`state_to_bytes`].
#[inline]
pub fn bytes_to_state(bytes: [u8; 4]) -> u32 {
    u32::from_ne_bytes(bytes)
}

/// Pack many `(id, props)` pairs into states in one pass, writing directly
/// into a pre-sized buffer instead of growing a `Vec` one `push` at a time —
/// the bulk path used when rehydrating a whole section's block-state array
/// from a decoded palette.
pub fn pack_states_batch(pairs: &[(u16, u32)]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(pairs.len());
    let ptr = out.as_mut_ptr();
    for (i, &(id, props)) in pairs.iter().enumerate() {
        let state = pack_state(id, props);
        // SAFETY: `ptr` comes from `Vec::with_capacity(pairs.len())`, so it
        // has room for `pairs.len()` elements; `i` ranges over
        // `0..pairs.len()` (the enumeration of `pairs`), so `ptr.add(i)`
        // stays within that reserved capacity, and this loop writes each
        // index in `0..pairs.len()` exactly once before `set_len` runs.
        unsafe {
            ptr.add(i).write(state);
        }
    }
    // SAFETY: the loop above wrote every index `0..pairs.len()` exactly
    // once, so `out`'s first `pairs.len()` elements are all initialized,
    // matching the length set below, which does not exceed the reserved
    // capacity.
    unsafe {
        out.set_len(pairs.len());
    }
    out
}

/// A small fluent builder for packed states, so call sites read as a list
/// of named property assignments instead of raw shifts and masks.
#[derive(Debug, Clone, Copy)]
pub struct StateBuilder {
    id: u16,
    props: u32,
}

impl StateBuilder {
    /// Start building a state for the given block id.
    pub fn new(id: u16) -> Self {
        StateBuilder { id, props: 0 }
    }

    /// Set a `width`-bit field at bit offset `offset` (within the property
    /// bits, i.e. offset 0 is the first bit after the id) to `value`.
    /// Panics if the field would spill past [`PROP_BITS`] or `value` does
    /// not fit in `width` bits.
    pub fn with_field(mut self, offset: u32, width: u32, value: u32) -> Self {
        assert!(offset + width <= PROP_BITS, "property field out of range");
        assert!(value < (1u32 << width), "value does not fit in field width");
        let mask = ((1u32 << width) - 1) << offset;
        self.props = (self.props & !mask) | (value << offset);
        self
    }

    /// Finish building, producing the packed `u32` state.
    pub fn build(self) -> u32 {
        pack_state(self.id, self.props)
    }
}

/// A compact, append-only registry mapping block names to numeric ids
/// (assigned sequentially on first registration) with a reverse lookup.
#[derive(Debug, Default)]
pub struct BlockRegistry {
    id_to_name: Vec<String>,
    name_to_id: HashMap<String, u16>,
}

impl BlockRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        BlockRegistry {
            id_to_name: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Register `name`, returning its id. Calling this again with the same
    /// name returns the same id rather than allocating a new one.
    pub fn register(&mut self, name: &str) -> u16 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.id_to_name.len() as u16;
        assert!(id <= MAX_BLOCK_ID, "block registry exceeded its id space");
        self.id_to_name.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Look up the id already assigned to `name`, without registering it.
    pub fn id_of(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    /// Look up the name assigned to `id`, if any.
    pub fn name_of(&self, id: u16) -> Option<&str> {
        let idx = id as usize;
        if idx >= self.id_to_name.len() {
            return None;
        }
        // SAFETY: the bounds check above guarantees `idx < self.id_to_name.len()`.
        Some(unsafe { self.id_to_name.get_unchecked(idx) })
    }

    /// Register `name` (if needed) and pack it with `props` into a state.
    pub fn pack(&mut self, name: &str, props: u32) -> u32 {
        let id = self.register(name);
        pack_state(id, props)
    }

    /// Split a packed state back into its registered name and properties,
    /// or `None` if the state's id was never registered here.
    pub fn unpack(&self, state: u32) -> Option<(&str, u32)> {
        let name = self.name_of(state_id(state))?;
        Some((name, state_props(state)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_state_round_trips() {
        let state = pack_state(42, 777);
        assert_eq!(state_id(state), 42);
        assert_eq!(state_props(state), 777);
    }

    #[test]
    fn pack_state_panics_on_id_overflow() {
        let result = std::panic::catch_unwind(|| pack_state(MAX_BLOCK_ID + 1, 0));
        assert!(result.is_err());
    }

    #[test]
    fn state_builder_matches_manual_bit_packing() {
        let built = StateBuilder::new(5)
            .with_field(0, 2, 3) // "facing" = 3
            .with_field(2, 1, 1) // "powered" = true
            .build();
        let expected_props = 3 | (1 << 2);
        assert_eq!(built, pack_state(5, expected_props));
        assert_eq!(state_id(built), 5);
        assert_eq!(state_props(built), expected_props);
    }

    #[test]
    fn state_bytes_round_trip_and_batch_matches_scalar_pack() {
        for state in [0u32, 1, 4095, 0xDEAD_BEEF, u32::MAX] {
            assert_eq!(bytes_to_state(state_to_bytes(state)), state);
        }
        let pairs = [(1u16, 0u32), (42, 7), (MAX_BLOCK_ID, MAX_PROPS)];
        let batch = pack_states_batch(&pairs);
        for (i, &(id, props)) in pairs.iter().enumerate() {
            assert_eq!(batch[i], pack_state(id, props));
        }
    }

    #[test]
    fn registry_dedups_repeated_names_and_assigns_fresh_ids_otherwise() {
        let mut reg = BlockRegistry::new();
        let stone1 = reg.register("stone");
        let dirt = reg.register("dirt");
        let stone2 = reg.register("stone");
        assert_eq!(stone1, stone2);
        assert_ne!(stone1, dirt);
        assert_eq!(reg.id_of("stone"), Some(stone1));
        assert_eq!(reg.id_of("unknown"), None);
    }

    #[test]
    fn registry_pack_unpack_round_trips_name_and_props() {
        let mut reg = BlockRegistry::new();
        let state = reg.pack("oak_log", 9);
        assert_eq!(reg.unpack(state), Some(("oak_log", 9)));
    }

    #[test]
    fn name_of_returns_none_for_unregistered_id() {
        let reg = BlockRegistry::new();
        assert_eq!(reg.name_of(0), None);
        assert_eq!(reg.name_of(MAX_BLOCK_ID), None);
    }
}

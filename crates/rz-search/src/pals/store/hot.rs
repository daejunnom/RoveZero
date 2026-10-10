//! Bounded hot slots retain original serial IDs; a freed slot never aliases one.
use super::StoreError;
use std::collections::HashMap;

#[derive(Debug)]
pub(super) struct HotRecords<T> {
    values: Vec<Option<(usize, T)>>,
    pub(super) index: HashMap<usize, usize>,
    next: usize,
    #[cfg(test)]
    fail_reservations: usize,
}

impl<T> HotRecords<T> {
    pub(super) fn new() -> Self {
        Self {
            values: Vec::new(),
            index: HashMap::new(),
            next: 0,
            #[cfg(test)]
            fail_reservations: 0,
        }
    }
    pub(super) fn len(&self) -> usize {
        self.index.len()
    }
    pub(super) fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
    pub(super) fn next_id(&self) -> usize {
        self.next
    }
    /// Requested backing capacity only. Hash table allocator internals and any
    /// shared allocations inside T are deliberately excluded.
    pub(super) fn owned_capacity_bytes(&self) -> Option<u64> {
        self.values
            .capacity()
            .checked_mul(std::mem::size_of::<Option<(usize, T)>>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }
    pub(super) fn reserve(&mut self, additional: usize) -> Result<(), StoreError> {
        #[cfg(test)]
        if self.fail_reservations != 0 {
            self.fail_reservations -= 1;
            return Err(StoreError::Capacity("injected hot allocation"));
        }
        self.next
            .checked_add(additional)
            .ok_or(StoreError::RevisionExhausted)?;
        let empty = self.values.len() - self.index.len();
        self.values
            .try_reserve_exact(additional.saturating_sub(empty))
            .map_err(|_| StoreError::Capacity("hot record allocation"))?;
        self.index
            .try_reserve(additional)
            .map_err(|_| StoreError::Capacity("hot record index allocation"))
    }
    #[cfg(test)]
    pub(super) fn fail_next_reservations(&mut self, count: usize) {
        self.fail_reservations = count;
    }
    fn put(&mut self, id: usize, value: T) {
        let slot = if let Some(slot) = self.values.iter().position(Option::is_none) {
            self.values[slot] = Some((id, value));
            slot
        } else {
            let slot = self.values.len();
            self.values.push(Some((id, value)));
            slot
        };
        self.index.insert(id, slot);
    }
    pub(super) fn push(&mut self, value: T) -> Result<(), StoreError> {
        self.reserve(1)?;
        self.put(self.next, value);
        self.next += 1;
        Ok(())
    }
    pub(super) fn get(&self, id: usize) -> Option<&T> {
        self.index
            .get(&id)
            .and_then(|slot| self.values[*slot].as_ref())
            .map(|(_, value)| value)
    }
    pub(super) fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        let slot = *self.index.get(&id)?;
        self.values[slot].as_mut().map(|(_, value)| value)
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &T> {
        self.values
            .iter()
            .filter_map(|value| value.as_ref().map(|(_, value)| value))
    }
    pub(super) fn entries(&self) -> impl Iterator<Item = (usize, &T)> {
        self.values
            .iter()
            .filter_map(|value| value.as_ref().map(|(id, value)| (*id, value)))
    }
    pub(super) fn remove(&mut self, id: usize) -> Option<T> {
        let slot = self.index.remove(&id)?;
        self.values[slot].take().map(|(_, value)| value)
    }
    pub(super) fn compact(&mut self) {
        self.values.retain(Option::is_some);
        for (slot, value) in self.values.iter().enumerate() {
            let id = value.as_ref().expect("retained live slot").0;
            *self.index.get_mut(&id).expect("retained live index") = slot;
        }
        self.values.shrink_to_fit();
        self.index.shrink_to_fit();
    }
    pub(super) fn restore(&mut self, id: usize, value: T) -> Result<(), StoreError> {
        if id >= self.next || self.get(id).is_some() {
            return Err(StoreError::InvalidHandle("archive serial"));
        }
        self.put(id, value);
        Ok(())
    }
}
impl<T> std::ops::Index<usize> for HotRecords<T> {
    type Output = T;
    fn index(&self, id: usize) -> &T {
        self.get(id).expect("live internal serial")
    }
}
impl<T> std::ops::IndexMut<usize> for HotRecords<T> {
    fn index_mut(&mut self, id: usize) -> &mut T {
        self.get_mut(id).expect("live internal serial")
    }
}

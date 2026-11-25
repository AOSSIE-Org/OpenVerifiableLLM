/-!
# Verifiable exclusion: skipping the removed targets

A fork records the removed article as target-id ranges `[start, count]` (`excluded` in
`simple/ovl.py`, sorted by start). Training then takes its next target from

```python
def next_target(cursor, ranges):
    for s, n in ranges:
        if s <= cursor < s + n:
            cursor = s + n
    return cursor
```

The three theorems say that, for ranges sorted by start, `next_target` skips exactly the excluded
targets: it never moves back (`le_nextTarget`), every target it skips is excluded
(`skipped_excluded`), and the target it returns is not excluded (`nextTarget_not_excluded`).
So the fork trains on every target the original run would, minus the recorded ranges.
-/

namespace OVL.Exclusion

/-- A range `(start, count)` holds target `t`. -/
def InRange (r : Nat × Nat) (t : Nat) : Prop := r.1 ≤ t ∧ t < r.1 + r.2

instance (r : Nat × Nat) (t : Nat) : Decidable (InRange r t) := by
  unfold InRange; exact inferInstance

/-- `next_target` from `simple/ovl.py`, line for line. -/
def nextTarget (cursor : Nat) : List (Nat × Nat) → Nat
  | [] => cursor
  | r :: rs => nextTarget (if InRange r cursor then r.1 + r.2 else cursor) rs

/-- Sorted by start, as `excluded` returns them. -/
def SortedByStart (rs : List (Nat × Nat)) : Prop := rs.Pairwise fun a b => a.1 ≤ b.1

theorem le_nextTarget (c : Nat) (rs : List (Nat × Nat)) : c ≤ nextTarget c rs := by
  induction rs generalizing c with
  | nil => simp [nextTarget]
  | cons r rs ih =>
    simp only [nextTarget]
    split
    · rename_i h; exact Nat.le_trans (Nat.le_of_lt h.2) (ih _)
    · exact ih _

/-- Every target the cursor jumps over is in a recorded range. -/
theorem skipped_excluded (c : Nat) (rs : List (Nat × Nat)) (t : Nat)
    (h₁ : c ≤ t) (h₂ : t < nextTarget c rs) : ∃ r ∈ rs, InRange r t := by
  induction rs generalizing c with
  | nil => simp [nextTarget] at h₂; omega
  | cons r rs ih =>
    simp only [nextTarget] at h₂
    split at h₂
    · rename_i hin
      by_cases ht : t < r.1 + r.2
      · exact ⟨r, by simp, ⟨by unfold InRange at hin; omega, ht⟩⟩
      · obtain ⟨r', hr', h'⟩ := ih _ (by omega) h₂
        exact ⟨r', by simp [hr'], h'⟩
    · obtain ⟨r', hr', h'⟩ := ih c h₁ h₂
      exact ⟨r', by simp [hr'], h'⟩

/-- A cursor before every range start does not move. -/
theorem nextTarget_before (c : Nat) (rs : List (Nat × Nat)) (h : ∀ r ∈ rs, c < r.1) :
    nextTarget c rs = c := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have hr : ¬ InRange r c := by
      have := h r (by simp); unfold InRange; omega
    simp only [nextTarget, hr, ite_false]
    exact ih (fun r' hr' => h r' (by simp [hr']))

/-- The target `next_target` returns is in no recorded range. -/
theorem nextTarget_not_excluded (c : Nat) (rs : List (Nat × Nat)) (hs : SortedByStart rs) :
    ∀ r ∈ rs, ¬ InRange r (nextTarget c rs) := by
  induction rs generalizing c with
  | nil => simp
  | cons r rs ih =>
    have hs' : SortedByStart rs := (List.pairwise_cons.mp hs).2
    have hle : ∀ r' ∈ rs, r.1 ≤ r'.1 := (List.pairwise_cons.mp hs).1
    intro r' hr'
    simp only [nextTarget]
    rcases List.mem_cons.mp hr' with rfl | hmem
    · -- the first range
      split
      · have := le_nextTarget (r'.1 + r'.2) rs
        unfold InRange; omega
      · rename_i hin
        unfold InRange at hin
        by_cases hc : c < r'.1
        · rw [nextTarget_before c rs (fun x hx => Nat.lt_of_lt_of_le hc (hle x hx))]
          unfold InRange; omega
        · have := le_nextTarget c rs
          unfold InRange; omega
    · exact ih _ hs' r' hmem

end OVL.Exclusion

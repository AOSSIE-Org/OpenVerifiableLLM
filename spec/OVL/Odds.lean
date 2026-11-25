/-!
# Detection odds

One segment out of `N` is wrong. `M` independent checkers each recompute one segment, chosen
uniformly at random. All `N ^ M` choices are equally likely, and the wrong segment is missed by
exactly the `(N - 1) ^ M` choices that never pick it (`count_miss`). So the probability of
missing it is `((N - 1) / N) ^ M`, as the README states.

For the released run, `N = 37`. `odds_95` and `odds_99` check the README's numbers: 110 checks
catch the wrong segment with at least 95% probability and 109 do not; 169 checks give 99% and
168 do not.

This assumes the choice is independent of the trainer. A checker who lets the trainer pick the
segment gets no guarantee from this.
-/

namespace OVL.Odds

/-- Every sequence of `m` choices from `vs`. -/
def seqs (vs : List Nat) : Nat → List (List Nat)
  | 0 => [[]]
  | m + 1 => vs.flatMap fun k => (seqs vs m).map (k :: ·)

theorem length_flatMap_const (vs : List Nat) (f : Nat → List (List Nat)) (c : Nat)
    (h : ∀ k, (f k).length = c) : (vs.flatMap f).length = vs.length * c := by
  induction vs with
  | nil => simp
  | cons v vs ih => simp [List.flatMap_cons, h, ih, Nat.succ_mul, Nat.add_comm]

theorem length_seqs (vs : List Nat) (m : Nat) : (seqs vs m).length = vs.length ^ m := by
  induction m with
  | zero => rfl
  | succ m ih =>
    simp only [seqs]
    rw [length_flatMap_const vs _ (vs.length ^ m) (fun k => by simp [ih]), Nat.pow_succ, Nat.mul_comm]

/-- A choice sequence misses `b` when it never picks `b`. -/
def misses (b : Nat) (l : List Nat) : Bool := !l.contains b

theorem filter_cons_map (b k : Nat) (S S' : List (List Nat)) (hS : S.filter (misses b) = S') :
    (S.map (k :: ·)).filter (misses b) = if k == b then [] else S'.map (k :: ·) := by
  subst hS
  rw [List.filter_map]
  by_cases hk : k = b
  · subst hk; simp [misses, Function.comp_def]
  · have : (fun l : List Nat => misses b (k :: l)) = misses b := by
      funext l; simp [misses, Ne.symm hk]
    simp [Function.comp_def, this, hk]

theorem filter_outer (b : Nat) (S S' : List (List Nat)) (hS : S.filter (misses b) = S') :
    ∀ ws : List Nat, (ws.flatMap fun k => S.map (k :: ·)).filter (misses b)
      = (ws.filter (· != b)).flatMap fun k => S'.map (k :: ·)
  | [] => rfl
  | w :: ws => by
    rw [List.flatMap_cons, List.filter_append, filter_cons_map b w S S' hS, filter_outer b S S' hS ws]
    by_cases hw : w = b
    · subst hw; simp
    · simp [hw, List.flatMap_cons]

/-- The choices that never pick `b` are exactly the choices made from the other segments. -/
theorem miss_eq (vs : List Nat) (b : Nat) :
    ∀ m, (seqs vs m).filter (misses b) = seqs (vs.filter (· != b)) m
  | 0 => rfl
  | m + 1 => filter_outer b _ _ (miss_eq vs b m) vs

/-- The segments of a run with `N` segments. -/
def segs (N : Nat) : List Nat := List.range N

theorem length_others (N b : Nat) (hb : b < N) : ((segs N).filter (· != b)).length = N - 1 := by
  induction N with
  | zero => omega
  | succ N ih =>
    simp only [segs] at ih ⊢
    rw [List.range_succ, List.filter_append, List.length_append]
    by_cases h : b = N
    · subst h
      have : (List.range b).filter (· != b) = List.range b :=
        List.filter_eq_self.mpr (fun a ha => by simp at ha; simp; omega)
      simp [this]
    · have : b < N := by omega
      rw [ih this]
      simp [Ne.symm h]
      omega

/-- Out of the `N ^ M` equally likely choices, exactly `(N - 1) ^ M` miss the wrong segment `b`. -/
theorem count_miss (N M b : Nat) (hb : b < N) :
    (seqs (segs N) M).length = N ^ M ∧
    ((seqs (segs N) M).filter (misses b)).length = (N - 1) ^ M := by
  constructor
  · simp [length_seqs, segs]
  · rw [miss_eq, length_seqs, length_others N b hb]

/-- `M` checks of a run with `N` segments miss a wrong segment with probability at most `1 / k`:
`(N - 1) ^ M / N ^ M ≤ 1 / k`. -/
def MissAtMost (N M k : Nat) : Prop := k * (N - 1) ^ M ≤ N ^ M

instance (N M k : Nat) : Decidable (MissAtMost N M k) := by unfold MissAtMost; exact inferInstance

/-- README: 110 random independent checks give 95% (miss at most 1/20); 109 do not. -/
theorem odds_95 : MissAtMost 37 110 20 ∧ ¬ MissAtMost 37 109 20 := by decide

/-- README: 169 random independent checks give 99% (miss at most 1/100); 168 do not. -/
theorem odds_99 : MissAtMost 37 169 100 ∧ ¬ MissAtMost 37 168 100 := by decide

end OVL.Odds

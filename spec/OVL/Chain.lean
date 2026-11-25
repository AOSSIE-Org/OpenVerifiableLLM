/-!
# Chain integrity

`simple/ovl.py` (`read_chain`, `cmd_verify`) accepts a run when:

* entry 0 has `previous = None`, and entry `i > 0` has
  `previous = sha256(json.dumps(entry[i-1], sort_keys=True))`;
* the number of entries and `sha256(json.dumps(last entry, sort_keys=True))` equal the
  `chain_entries` and `chain_head_sha256` of the timestamped `anchor.json`.

The theorem `anchor_binds_chain`: two chains that both pass these checks against the same anchor
are equal, or the two chains together contain a hash collision. Nothing else is assumed about the
hash. `enc` (canonical JSON with sorted keys) must be injective; that is the hypothesis `hEnc`.

Chains are written oldest first, as in `chain.jsonl`.
-/

namespace OVL.Chain

variable {E B D : Type}

/-- `enc` never maps two different entries to the same bytes. -/
def Injective (f : E → B) : Prop := ∀ a b, f a = f b → a = b

/-- Two different inputs with the same hash. -/
def Collision (H : B → D) : Prop := ∃ x y, x ≠ y ∧ H x = H y

/-- The link check of `read_chain`, on a chain written newest first. -/
def LinkedRev (enc : E → B) (H : B → D) (prev : E → Option D) : List E → Prop
  | [] => True
  | [e] => prev e = none
  | e :: f :: rest => prev e = some (H (enc f)) ∧ LinkedRev enc H prev (f :: rest)

/-- The link check of `read_chain`, on a chain written oldest first. -/
def Linked (enc : E → B) (H : B → D) (prev : E → Option D) (c : List E) : Prop :=
  LinkedRev enc H prev c.reverse

/-- What `anchor.json` records: the number of entries and the hash of the last entry. -/
def anchorOf (enc : E → B) (H : B → D) (c : List E) : Option (Nat × D) :=
  c.getLast?.map fun e => (c.length, H (enc e))

theorem linkedRev_binds (enc : E → B) (H : B → D) (prev : E → Option D) (hEnc : Injective enc) :
    ∀ (r₁ r₂ : List E) (e₁ e₂ : E),
      LinkedRev enc H prev (e₁ :: r₁) → LinkedRev enc H prev (e₂ :: r₂) →
      r₁.length = r₂.length → H (enc e₁) = H (enc e₂) →
      e₁ :: r₁ = e₂ :: r₂ ∨ Collision H := by
  intro r₁
  induction r₁ with
  | nil =>
    intro r₂ e₁ e₂ _ _ hlen hh
    cases r₂ with
    | cons _ _ => simp at hlen
    | nil =>
      by_cases he : enc e₁ = enc e₂
      · exact .inl (by rw [hEnc _ _ he])
      · exact .inr ⟨_, _, he, hh⟩
  | cons f₁ r₁ ih =>
    intro r₂ e₁ e₂ h₁ h₂ hlen hh
    cases r₂ with
    | nil => simp at hlen
    | cons f₂ r₂ =>
      by_cases he : enc e₁ = enc e₂
      · have hee : e₁ = e₂ := hEnc _ _ he
        subst hee
        have hf : H (enc f₁) = H (enc f₂) := by
          have := h₁.1.symm.trans h₂.1
          simpa using this
        rcases ih r₂ f₁ f₂ h₁.2 h₂.2 (by simpa using hlen) hf with heq | hc
        · exact .inl (by rw [heq])
        · exact .inr hc
      · exact .inr ⟨_, _, he, hh⟩

/-- Two chains that pass the link check and match the same anchor are the same chain,
or they contain a hash collision. -/
theorem anchor_binds_chain (enc : E → B) (H : B → D) (prev : E → Option D) (hEnc : Injective enc)
    (c₁ c₂ : List E) (h₁ : Linked enc H prev c₁) (h₂ : Linked enc H prev c₂)
    (ha : anchorOf enc H c₁ = anchorOf enc H c₂) (hne : c₁ ≠ []) :
    c₁ = c₂ ∨ Collision H := by
  unfold Linked at h₁ h₂
  unfold anchorOf at ha
  cases hr₁ : c₁.reverse with
  | nil => exact absurd (List.reverse_eq_nil_iff.mp hr₁) hne
  | cons e₁ r₁ =>
    cases hr₂ : c₂.reverse with
    | nil =>
      have : c₂ = [] := List.reverse_eq_nil_iff.mp hr₂
      subst this
      have : c₁.getLast? ≠ none := by simpa using hne
      simp_all
    | cons e₂ r₂ =>
      have g₁ : c₁.getLast? = some e₁ := by
        rw [← List.head?_reverse, hr₁]; rfl
      have g₂ : c₂.getLast? = some e₂ := by
        rw [← List.head?_reverse, hr₂]; rfl
      rw [g₁, g₂] at ha
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at ha
      have l₁ : c₁.length = r₁.length + 1 := by
        rw [← List.length_reverse, hr₁]; rfl
      have l₂ : c₂.length = r₂.length + 1 := by
        rw [← List.length_reverse, hr₂]; rfl
      rw [hr₁] at h₁
      rw [hr₂] at h₂
      rcases linkedRev_binds enc H prev hEnc r₁ r₂ e₁ e₂ h₁ h₂ (by omega) ha.2 with heq | hc
      · left
        rw [← List.reverse_reverse c₁, ← List.reverse_reverse c₂, hr₁, hr₂, heq]
      · exact .inr hc

end OVL.Chain

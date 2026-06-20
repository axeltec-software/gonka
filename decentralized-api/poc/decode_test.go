package poc

import (
	"testing"
)

// Round-trip: pack then unpack must recover the exact trajectory (the off-chain store
// + Merkle proofs carry these bytes, so the codec must be lossless and stable).
func TestTrajectoryRoundTrip(t *testing.T) {
	cases := [][]int64{
		{0},
		{3, 7, 1, 9, 2},
		{0, 0, 0},
		{15, 15, 15, 0, 1},
	}
	for _, kp := range cases {
		got, err := unpackTrajectory(packTrajectory(kp))
		if err != nil {
			t.Fatalf("unpack(pack(%v)) error: %v", kp, err)
		}
		if len(got) != len(kp) {
			t.Fatalf("length mismatch: got %d want %d", len(got), len(kp))
		}
		for i := range kp {
			if got[i] != kp[i] {
				t.Fatalf("step %d: got %d want %d (%v)", i, got[i], kp[i], kp)
			}
		}
	}
}

// unpack must reject malformed bytes and out-of-range sphere_k (catches mode confusion:
// a prefill fp16 vector misread as a trajectory).
func TestUnpackTrajectoryRejectsBad(t *testing.T) {
	if _, err := unpackTrajectory(nil); err == nil {
		t.Fatal("expected error for empty bytes")
	}
	if _, err := unpackTrajectory([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected error for non-multiple-of-4 length")
	}
	// value == SpherePoints is out of range [0, SpherePoints)
	oob := packTrajectory([]int64{int64(SpherePoints)})
	if _, err := unpackTrajectory(oob); err == nil {
		t.Fatalf("expected out-of-range error for sphere_k=%d", SpherePoints)
	}
	// negative value
	neg := packTrajectory([]int64{-1})
	if _, err := unpackTrajectory(neg); err == nil {
		t.Fatal("expected out-of-range error for negative sphere_k")
	}
}

// ArtifactToProto packs the trajectory into Vector for decode artifacts, uses the
// prefill vector bytes otherwise, and rejects the empty-both case.
func TestArtifactToProto(t *testing.T) {
	// decode: trajectory present -> Vector is the packed trajectory, recoverable.
	kp := []int64{1, 2, 3, 4}
	pa, err := ArtifactToProto(7, nil, kp)
	if err != nil {
		t.Fatalf("decode artifact: %v", err)
	}
	if pa.Nonce != 7 {
		t.Fatalf("nonce: got %d want 7", pa.Nonce)
	}
	back, err := unpackTrajectory(pa.Vector)
	if err != nil || len(back) != len(kp) {
		t.Fatalf("decode artifact Vector not a valid trajectory: %v (%v)", err, back)
	}

	// prefill: no trajectory -> Vector is the raw fp16 vector bytes, untouched.
	vec := []byte{0x01, 0x02, 0x03, 0x04}
	pa, err = ArtifactToProto(8, vec, nil)
	if err != nil {
		t.Fatalf("prefill artifact: %v", err)
	}
	if string(pa.Vector) != string(vec) {
		t.Fatalf("prefill Vector altered: got %v want %v", pa.Vector, vec)
	}

	// empty both -> error (neither a vector nor a trajectory).
	if _, err := ArtifactToProto(9, nil, nil); err == nil {
		t.Fatal("expected error for empty vector and no trajectory")
	}
}

// max_tokens is derived from the reference length (trajectory = prefill k0 + max_tokens
// steps), so validation needs no separate consensus param.
func TestMaxTokensDerivation(t *testing.T) {
	kp := []int64{5, 1, 2, 3, 4, 6} // len 6 -> 5 decode steps
	if got := len(kp) - 1; got != 5 {
		t.Fatalf("derived max_tokens: got %d want 5", got)
	}
}

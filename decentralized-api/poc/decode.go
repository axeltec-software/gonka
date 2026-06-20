package poc

import (
	"encoding/binary"
	"fmt"

	"github.com/productscience/inference/x/inference/types"
)

// SpherePoints is the codebook size; valid sphere_k values are [0, SpherePoints).
// Used only to sanity-check unpacked trajectories. Mirrors vllm/poc/sphere.py.
// (The decode-PoC enable flag + generation max_tokens live in mlnodeclient to avoid
// an import cycle — poc imports broker, so the gate can't live here for broker's use.)
const SpherePoints = 16

// packTrajectory encodes a sphere_k trajectory into the opaque PoCArtifactV2.Vector
// bytes (big-endian int32 per step). Decode artifacts leave vector_b64 empty, so the
// Vector field is free to carry the trajectory — no proto change. The bytes ride
// through the existing off-chain store + Merkle proofs, so the reference trajectory is
// tamper-evident for free. Pack (ingest) and unpack (validate) are both Go-side here.
func packTrajectory(kPoints []int64) []byte {
	buf := make([]byte, 4*len(kPoints))
	for i, k := range kPoints {
		binary.BigEndian.PutUint32(buf[i*4:], uint32(int32(k)))
	}
	return buf
}

// unpackTrajectory reverses packTrajectory. Returns an error if the bytes are not a
// whole number of int32s or any value is outside [0, SpherePoints) — catching mode
// confusion (e.g. a prefill vector misread as a trajectory) loudly instead of
// silently mis-scoring.
func unpackTrajectory(b []byte) ([]int64, error) {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil, fmt.Errorf("decode trajectory: bad length %d (not a multiple of 4)", len(b))
	}
	n := len(b) / 4
	out := make([]int64, n)
	for i := 0; i < n; i++ {
		v := int32(binary.BigEndian.Uint32(b[i*4:]))
		if v < 0 || int(v) >= SpherePoints {
			return nil, fmt.Errorf("decode trajectory: sphere_k %d out of range [0,%d)", v, SpherePoints)
		}
		out[i] = int64(v)
	}
	return out, nil
}

// ArtifactToProto converts an MLNode artifact to its off-chain proto form, packing
// the decode trajectory into Vector when present (decode) or using the prefill vector
// bytes otherwise. vectorBytes is the base64-decoded VectorB64 (prefill path).
func ArtifactToProto(nonce int64, vectorBytes []byte, kPoints []int64) (*types.PoCArtifactV2, error) {
	if len(kPoints) > 0 {
		return &types.PoCArtifactV2{Nonce: int32(nonce), Vector: packTrajectory(kPoints)}, nil
	}
	if len(vectorBytes) == 0 {
		return nil, fmt.Errorf("artifact %d: empty vector and no trajectory", nonce)
	}
	return &types.PoCArtifactV2{Nonce: int32(nonce), Vector: vectorBytes}, nil
}

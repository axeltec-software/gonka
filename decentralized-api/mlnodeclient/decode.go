package mlnodeclient

// Decode-PoC (sphere_k trajectory) configuration.
//
// These are BINARY-LEVEL constants, not chain params (intentionally no proto change):
// a value baked into the binary is consensus-consistent across nodes on the same
// release, so a prover and a validator interpret artifacts identically. Default OFF,
// so this code is dormant and the prefill PoC path is 100% unchanged until a future
// release flips PoCDecodeEnabled. (Same coordination model as a binary upgrade.)
//
// They live in mlnodeclient (imported by broker, poc, and the mlnode handler; it
// imports none of them) so all sites read the SAME value without an import cycle.
const (
	// PoCDecodeEnabled gates decode-PoC end to end. When false, generation runs
	// prefill (MaxTokens=0) and validation scores vectors — i.e. today's behavior.
	PoCDecodeEnabled = false
	// PoCDecodeMaxTokens is the decode trajectory length used during GENERATION.
	// Validation derives its own max_tokens from the reference length, so this only
	// drives mining. Network-wide constant -> all provers mine the same shape.
	PoCDecodeMaxTokens = 256
)

// PoCDecodeGenMaxTokens returns the generation max_tokens: PoCDecodeMaxTokens when
// decode-PoC is enabled, else 0 (prefill-only).
func PoCDecodeGenMaxTokens() int64 {
	if PoCDecodeEnabled {
		return PoCDecodeMaxTokens
	}
	return 0
}

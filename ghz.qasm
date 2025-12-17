OPENQASM 3.0;
include "stdgates.inc";

qubit[40] q;

// --- Cluster A (0,1,2): your original ---
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[0], q[2];
cx q[2], q[1];
cx q[1], q[0];

// --- Cluster B (3,4,5): your original ---
cx q[3], q[4];
cx q[4], q[5];
cx q[3], q[5];

// --- Add some single-qubit structure (helps layering variety) ---
h q[6]; h q[7]; h q[8];
h q[9]; h q[10]; h q[11];

// --- Cluster C (6-11): dense-ish local entanglement ---
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[6], q[8];
cx q[7], q[9];
cx q[8], q[10];
cx q[9], q[11];

// --- Cluster D (12-17): another local block ---
h q[12]; h q[13]; h q[14]; h q[15]; h q[16]; h q[17];
cx q[12], q[13];
cx q[13], q[14];
cx q[14], q[15];
cx q[15], q[16];
cx q[16], q[17];
cx q[12], q[15];
cx q[13], q[16];
cx q[14], q[17];

// --- Cluster E (18-23): local block with some triangles ---
h q[18]; h q[19]; h q[20]; h q[21]; h q[22]; h q[23];
cx q[18], q[19];
cx q[19], q[20];
cx q[20], q[18];
cx q[21], q[22];
cx q[22], q[23];
cx q[23], q[21];
cx q[18], q[21];
cx q[20], q[23];

// --- Cluster F (24-31): 8-qubit chain + chords ---
h q[24]; h q[25]; h q[26]; h q[27]; h q[28]; h q[29]; h q[30]; h q[31];
cx q[24], q[25];
cx q[25], q[26];
cx q[26], q[27];
cx q[27], q[28];
cx q[28], q[29];
cx q[29], q[30];
cx q[30], q[31];
cx q[24], q[26];
cx q[25], q[27];
cx q[26], q[28];
cx q[27], q[29];
cx q[28], q[30];
cx q[29], q[31];

// --- Light use of remaining qubits (32-39) as an "aux cluster" ---
h q[32]; h q[33]; h q[34]; h q[35]; h q[36]; h q[37]; h q[38]; h q[39];
cx q[32], q[33];
cx q[34], q[35];
cx q[36], q[37];
cx q[38], q[39];
cx q[32], q[34];
cx q[35], q[36];
cx q[37], q[38];

// --- Cross-cluster links (these create nontrivial cuts) ---
cx q[2],  q[3];
cx q[5],  q[6];
cx q[11], q[12];
cx q[17], q[18];
cx q[23], q[24];
cx q[31], q[32];

// --- Repeat cross links later to create "time dependence" in your layered graph ---
h q[0]; h q[3]; h q[6]; h q[12]; h q[18]; h q[24]; h q[32];
cx q[0],  q[6];
cx q[3],  q[9];
cx q[6],  q[12];
cx q[12], q[18];
cx q[18], q[24];
cx q[24], q[32];

// --- A few more repeated local interactions (increase edge counts across layers) ---
cx q[0], q[1];
cx q[1], q[2];
cx q[3], q[4];
cx q[4], q[5];
cx q[24], q[25];
cx q[25], q[26];
cx q[32], q[33];
cx q[33], q[34];

// --- Some extra cross links to prevent trivial partitioning ---
cx q[10], q[20];
cx q[8],  q[28];
cx q[14], q[30];
cx q[16], q[36];
cx q[22], q[38];

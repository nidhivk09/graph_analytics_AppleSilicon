// common.h: shared by every kernel file.
// The host prepends this file to each .metal source before runtime compilation,
// because newLibraryWithSource does not resolve local #includes.
#include <metal_stdlib>
using namespace metal;

// Marks an unreached vertex in BFS distance arrays (matches the host constant).
constant uint INF_DIST = 0xFFFFFFFFu;

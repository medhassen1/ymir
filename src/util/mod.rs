//! The engine's self-contained building blocks.
//!
//! Noise and pseudo-random generators for terrain, spatial math and traversal,
//! encoding and container primitives, and the voxel/rendering helpers the mesher
//! and lighter are built on. Everything here is dependency-free and unit-tested,
//! and nothing in this module reaches back into the region pipeline.

pub mod aabb;
pub mod ao;
pub mod arena;
pub mod binheap;
pub mod biometable;
pub mod bitpack;
pub mod bitset;
pub mod blockstate;
pub mod bounds2;
pub mod chunkcoord;
pub mod color;
pub mod crc32;
pub mod deltacode;
pub mod facemask;
pub mod fbm;
pub mod fnv;
pub mod frustum;
pub mod greedy;
pub mod gridwalk;
pub mod heightfield;
pub mod interner;
pub mod ivec3;
pub mod lerp;
pub mod lightmap;
pub mod lrucache;
pub mod lz;
pub mod mat4;
pub mod morton;
pub mod murmur3;
pub mod normals;
pub mod octree;
pub mod pcg32;
pub mod perlin;
pub mod plane;
pub mod quantize;
pub mod quat;
pub mod ray;
pub mod ringbuf;
pub mod rle;
pub mod simplex;
pub mod siphash;
pub mod skylight;
pub mod slotmap;
pub mod smallvec;
pub mod spatialhash;
pub mod splitmix64;
pub mod tangent;
pub mod tickpriority;
pub mod transform;
pub mod valuenoise;
pub mod varint;
pub mod vec3;
pub mod worley;
pub mod xoroshiro;
pub mod xxhash32;
pub mod zigzag;

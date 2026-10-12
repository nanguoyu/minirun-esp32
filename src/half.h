#ifndef LFM_HALF_H
#define LFM_HALF_H
#include <stdint.h>
#include <string.h>
/* IEEE binary16 storage. Arithmetic remains binary32. */
static inline float lfm_half_expand(uint16_t h) {
    uint32_t bits = ((uint32_t)h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 31, mant = h & 1023;
    if (e == 31) bits |= 0x7f800000 | (mant << 13);
    else if (e) bits |= ((e + 112) << 23) | (mant << 13);
    else if (mant) {
        int exponent = -14;
        while (!(mant & 1024)) { mant <<= 1; exponent--; }
        bits |= ((uint32_t)(exponent + 127) << 23) | ((mant & 1023) << 13);
    }
    float value; memcpy(&value, &bits, sizeof(value)); return value;
}
/* Round to nearest, ties to even, including gradual underflow and overflow. */
static inline uint16_t lfm_half_pack(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000, mant = bits & 0x7fffff;
    int exponent = (int)((bits >> 23) & 255) - 127;
    if (exponent == 128)
        return (uint16_t)(sign | 0x7c00 | (mant ? (0x200 | (mant >> 13)) : 0));
    if (exponent > 15) return (uint16_t)(sign | 0x7c00);
    if (exponent < -25) return (uint16_t)sign;
    if (exponent < -14) {
        mant |= 0x800000;
        unsigned shift = (unsigned)(-exponent - 1);
        uint32_t q = mant >> shift, rem = mant & ((1u << shift) - 1);
        uint32_t midpoint = 1u << (shift - 1);
        if (rem > midpoint || (rem == midpoint && (q & 1))) q++;
        return (uint16_t)(sign | q);
    }
    uint32_t rounded = (mant + 0xfff + ((mant >> 13) & 1)) >> 13;
    return (uint16_t)(sign | (((uint32_t)(exponent + 15) << 10) + rounded));
}
#endif

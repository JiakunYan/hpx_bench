#include <cstdint>

namespace hpcc
{
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

typedef struct params_t {
  int64_t LocalTableSize; /* local size of the table may be rounded up >=
                             MinLocalTableSize */
  int64_t ProcNumUpdates; /* usually 4 times the local size except for
                             time-bound runs */

  uint64_t logTableSize;      /* it is an unsigned 64-bit value to type-promote
                                 expressions */
  uint64_t TableSize;         /* always power of 2 */
  uint64_t MinLocalTableSize; /* TableSize/NumProcs */
  uint64_t GlobalStartMyProc; /* first global index of the global table stored
                                 locally */
  uint64_t Top; /* global indices below 'Top' are asigned in MinLocalTableSize+1
                 blocks; above 'Top' -- in MinLocalTableSize blocks */
  int logNumProcs, NumProcs, MyProc;
  int Remainder; /* TableSize % NumProcs */
} params_t;

uint64_t starts(int64_t n)
{
  int i, j;
  uint64_t m2[64];
  uint64_t temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1) break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1) temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1) ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY : 0);
  }

  return ran;
}

// Function to generate the next random value
inline uint64_t generate_next_random(uint64_t random)
{
  return (random << 1) ^ ((int64_t)random < 0 ? POLY : 0);
}

}  // namespace hpcc
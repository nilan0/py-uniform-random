"""
Uniform pseudorandom number generator meant to emulate Valve's original implementation in C++.
"""

import math


class UniformRandom:
    """
    Generates pseudorandom uniformly distributed ints and floats in user-specified ranges.
    """

    __NTAB: int = 32

    # Values used by Park and Miller as shown in Numerical Recipes for C
    __IA: int = 16807  # 7**5
    __IM: int = 2**31 - 1

    # Values used by Schrage as shown in Numerical Recipes for C
    __IQ: int = 127773
    __IR: int = 2836

    __NDIV: int = (__IM - 1) // __NTAB + 1
    __AM: float = 1.0 / __IM
    __EPS: float = 1.2e-7
    __RNMX: float = 1.0 - __EPS

    __MAX_RANDOM_RANGE: int = 2**31 - 1


    def __init__(self, seed: int = 0) -> None:
        self.__idum = 0
        self.__iy = 0
        self.__iv = [0] * self.__NTAB
        self.set_seed(seed)


    def set_seed(self, seed: int) -> None:
        """Set the seed that determines the starting point for the pseudorandom number generator.

        Args:
            seed: the seed value to set for the generator

        Returns:
            None
        """
        self.__idum = -abs(seed)
        self.__iy = 0


    def __next_number(self) -> int:
        if self.__idum <= 0:
            if -self.__idum < 1:  # Prevent idum = 0
                self.__idum = 1
            else:
                self.__idum = -self.__idum

            for j in range(self.__NTAB + 7, -1, -1):  # Load shuffle table
                k = self.__idum // self.__IQ
                self.__idum = self.__IA * (self.__idum - k * self.__IQ) - self.__IR * k
                if self.__idum < 0:
                    self.__idum += self.__IM
                if j < self.__NTAB:
                    self.__iv[j] = self.__idum

            self.__iy = self.__iv[0]

        k = self.__idum // self.__IQ

        # Schrage's method for multiplying two 32-bit integers modulo a 32-bit
        # constant without any intermediate values larger than 32 bits.
        # Used here to avoid overflow when computing idum = (IA * idum) % IM
        self.__idum = self.__IA * (self.__idum - k * self.__IQ) - self.__IR * k
        if self.__idum < 0:
            self.__idum += self.__IM

        # Output previously stored value and replace it in the shuffle table.
        # The random number iy is used to choose a random element in iv,
        # which is the output and the next iy.
        j = self.__iy // self.__NDIV
        self.__iy = self.__iv[j]
        self.__iv[j] = self.__idum

        return self.__iy


    def next_float(self, lo: float = 0.0, hi: float = 1.0, exp: float = 1.0) -> float:
        """Get the next pseudorandom float in a sequence based on the seed,
        uniformly distributed in [lo, hi), optionally raised to the power of exp.
        
        Args:
            lo -- lower bound (inclusive, default 0.0)
            hi -- upper bound (exclusive, default 1.0)
            exp -- exponent (default, 1.0)
        
        Returns:
            A pseudorandom float in [lo, hi) raised to exp
        """

        # Float in [0, 1)
        fl = self.__AM * self.__next_number()
        if fl > self.__RNMX:
            fl = self.__RNMX

        if exp != 1.0:
            fl = math.pow(fl, exp)

        return fl * (hi - lo) + lo # Float in [lo, hi) (raised to exp)


    def next_int(self, lo: int, hi: int) -> int:
        """Get the next pseudorandom int in a sequence based on the seed,
        uniformly distributed in [lo, hi].
        
        Args:
            lo: lower bound (inclusive)
            hi: upper bound (inclusive)

        Returns:
            A pseudorandom int in [lo, hi]
        """

        random_range: int = hi - lo + 1

        # Make sure range is valid
        if (hi <= lo) or (self.__MAX_RANDOM_RANGE < random_range - 1):
            return lo

        # Map [0, MAX_RANDOM_RANGE] to the specified range [0, hi - lo]
        max_acceptable = self.__MAX_RANDOM_RANGE - ((self.__MAX_RANDOM_RANGE + 1) % random_range)

        while True:
            n = self.__next_number()
            if n <= max_acceptable:
                break

        return n % random_range + lo # Int in [lo, hi]
    
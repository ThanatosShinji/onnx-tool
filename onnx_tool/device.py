# Device specifications for latency estimation.
# All compute values are in GFLOPS (10^9 FLOPS) / GOPS (10^9 OPS).
# Bandwidth is in GB/s (10^9 bytes/s).
# FP16 field also serves as BF16 compute.
# INT8 is typically 2x FP16 (for GPUs with tensor cores).
# FP32 is typically FP16 / 2 or FP16 / 8 depending on architecture.
Devices = {
    'Core-13900':
        {
            'FP32': 1250,
            'INT8': 5000,
            'Bandwidth': 75.5,
        },
    'Ultra-155H': {
        'FP32': 4000,
        'FP16': 8000,
        'INT8': 16000,
        'Bandwidth': 90,
    },
    'Ultra-358H': {
        'FP32': 7500,
        'FP16': 60000,
        'INT8': 120000,
        'Bandwidth': 128,
    },
    'Arc-A750': {
        'FP32': 14700,
        'FP16': 117000,
        'INT8': 235000,
        'Bandwidth': 512,
    },
    'Arc-A770': {
        'FP32': 17200,
        'FP16': 138000,
        'INT8': 275000,
        'Bandwidth': 560,
    },
    'Arc-B70': {
        'FP32': 22900,
        'FP16': 183500,
        'INT8': 367000,
        'Bandwidth': 608,
    },
    'A100-40GB-PCIe': {
        'FP32': 19500,
        'FP16': 312000,
        'INT8': 624000,
        'Bandwidth': 1935,
    },
    'A30': {
        'FP32': 10300,
        'FP16': 165000,
        'INT8': 330000,
        'Bandwidth': 933,
    },
    'A40': {
        'FP32': 37400,
        'FP16': 149700,
        'INT8': 299300,
        'Bandwidth': 696,
    },
    'A800-40GB-PCIe': {
        'FP32': 19500,
        'FP16': 312000,
        'INT8': 624000,
        'Bandwidth': 1555,
    },
    'H100-PCIe': {
        'FP32': 48000,
        'FP16': 1600000,
        'INT8': 3200000,
        'Bandwidth': 2000,
    },
    'Gaudi2H': {
        'FP32': 11000,
        'FP16': 432000,
        'INT8': 865000,
        'Bandwidth': 2460,
    },
    'H20': {
        'FP32': 44000,
        'FP16': 148000,
        'INT8': 296000,
        'Bandwidth': 900,
    },
    'RTX-4090': {
        # Ada Lovelace AD102, 24GB GDDR6X, 512 Tensor Cores (4th gen)
        'FP32': 82600,
        'FP16': 165200,
        'INT8': 660600,
        'Bandwidth': 1008,
    },
    'RTX-5090': {
        # Blackwell GB202, 32GB GDDR7, 680 Tensor Cores (5th gen)
        'FP32': 104800,
        'FP16': 209500,
        'INT8': 1676000,
        'Bandwidth': 1792,
    },
}

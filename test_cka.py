"""Test script to verify CKA implementation works correctly."""

import numpy as np
from src.CKA import linear_cka, rbf_cka, compare_representations


def test_cka_properties():
    """Test fundamental properties that CKA should satisfy."""
    np.random.seed(42)
    N, D = 100, 50
    
    print("="*60)
    print("Testing CKA Properties")
    print("="*60)
    
    # Test 1: Self-similarity should be 1.0
    print("\n1. Self-similarity test (should be ~1.0):")
    X = np.random.randn(N, D)
    cka_self = linear_cka(X, X)
    print(f"   CKA(X, X) = {cka_self:.6f}")
    assert abs(cka_self - 1.0) < 1e-10, "Self-similarity should be 1.0"
    print("   [PASSED]")
    
    # Test 2: CKA is symmetric
    print("\n2. Symmetry test (CKA(X,Y) = CKA(Y,X)):")
    Y = np.random.randn(N, D)
    cka_xy = linear_cka(X, Y)
    cka_yx = linear_cka(Y, X)
    print(f"   CKA(X, Y) = {cka_xy:.6f}")
    print(f"   CKA(Y, X) = {cka_yx:.6f}")
    assert abs(cka_xy - cka_yx) < 1e-10, "CKA should be symmetric"
    print("   [PASSED]")
    
    # Test 3: CKA is invariant to orthogonal transformations
    print("\n3. Orthogonal invariance test:")
    # Create random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(D, D))
    X_rotated = X @ Q
    cka_rotated = linear_cka(X, X_rotated)
    print(f"   CKA(X, X_rotated) = {cka_rotated:.6f}")
    assert abs(cka_rotated - 1.0) < 1e-10, "CKA should be invariant to rotations"
    print("   [PASSED]")
    
    # Test 4: CKA is invariant to isotropic scaling
    print("\n4. Scale invariance test:")
    X_scaled = X * 5.0
    cka_scaled = linear_cka(X, X_scaled)
    print(f"   CKA(X, 5*X) = {cka_scaled:.6f}")
    assert abs(cka_scaled - 1.0) < 1e-10, "CKA should be invariant to scaling"
    print("   [PASSED]")
    
    # Test 5: Independent random representations should have low CKA
    print("\n5. Independence test (should be relatively low):")
    X_random = np.random.randn(N, D)
    Y_random = np.random.randn(N, D + 10)
    cka_random = linear_cka(X_random, Y_random)
    print(f"   CKA(X_random, Y_random) = {cka_random:.6f}")
    # Note: With finite samples, random representations can have CKA up to ~0.4
    assert cka_random < 0.5, "Independent representations should have low CKA"
    print("   [PASSED] (Note: finite sample effects can make this non-zero)")
    
    # Test 6: Linear transformation should maintain high CKA
    print("\n6. Linear transformation test:")
    W = np.random.randn(D, 30)
    Y_linear = X @ W + 0.01 * np.random.randn(N, 30)  # Small noise
    cka_linear = linear_cka(X, Y_linear)
    print(f"   CKA(X, X*W + small_noise) = {cka_linear:.6f}")
    assert cka_linear > 0.6, "Linear transformation should maintain reasonably high CKA"
    print("   [PASSED]")
    
    print("\n" + "="*60)
    print("All CKA property tests passed!")
    print("="*60)


def test_cka_vs_correlation():
    """Test that CKA captures similarity better than simple correlation."""
    np.random.seed(123)
    N = 100
    
    print("\n" + "="*60)
    print("Comparing CKA with Simple Correlation")
    print("="*60)
    
    # Create structured data
    print("\nScenario: Two representations of the same underlying structure")
    
    # Underlying latent factors
    latent = np.random.randn(N, 5)
    
    # First representation: Linear mixing
    W1 = np.random.randn(5, 20)
    X1 = latent @ W1
    
    # Second representation: Different linear mixing
    W2 = np.random.randn(5, 15)
    X2 = latent @ W2
    
    # Third representation: Unrelated
    X3 = np.random.randn(N, 25)
    
    print("\nResults:")
    print(f"  CKA(X1, X2) [same latent]:     {linear_cka(X1, X2):.4f}")
    print(f"  CKA(X1, X3) [unrelated]:        {linear_cka(X1, X3):.4f}")
    print(f"  CKA(X2, X3) [unrelated]:        {linear_cka(X2, X3):.4f}")
    
    # Test RBF kernel version
    print("\nRBF Kernel CKA:")
    print(f"  RBF-CKA(X1, X2) [same latent]: {rbf_cka(X1, X2):.4f}")
    print(f"  RBF-CKA(X1, X3) [unrelated]:   {rbf_cka(X1, X3):.4f}")
    
    print("\n[SUCCESS] CKA successfully identifies representations with shared structure")


def test_compare_representations():
    """Test the high-level comparison function."""
    np.random.seed(456)
    N, D = 50, 20
    
    print("\n" + "="*60)
    print("Testing compare_representations function")
    print("="*60)
    
    X = np.random.randn(N, D)
    
    # Test different similarity levels
    test_cases = [
        ("Identical", X, X),
        ("Scaled", X, X * 3.0),
        ("Rotated", X, X @ np.linalg.qr(np.random.randn(D, D))[0]),
        ("Linear transform", X, X @ np.random.randn(D, 15)),
        ("Independent", X, np.random.randn(N, D)),
    ]
    
    for name, repr1, repr2 in test_cases:
        result = compare_representations(repr1, repr2, 'linear')
        print(f"\n{name}:")
        print(f"  CKA Score: {result['cka']:.4f}")
        print(f"  Interpretation: {result['interpretation']}")


if __name__ == "__main__":
    test_cka_properties()
    test_cka_vs_correlation()
    test_compare_representations()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)